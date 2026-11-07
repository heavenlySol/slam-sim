"""Self-contained 2-D occupancy-grid SLAM-style simulator.

Uses simulated 2-D LiDAR, noisy differential-drive odometry, scan-to-map pose
correction against the accumulated occupancy grid, and inverse sensor-model updates.
This is a pedagogical simulation, not a substitute for a tested SLAM stack such as
Cartographer, slam_toolbox, or GMapping on a physical robot.
"""
import math
import heapq
import numpy as np
import cv2
from collections import deque

N = 300
RES = 0.1
HALF = 15.0
LIDAR_MAX = 6.0
RAY_COUNT = 120
# Warehouse lane-centering assumptions: 50 cm rover in a nominal 200 cm aisle.
ROVER_WIDTH_M = 0.50
AISLE_WIDTH_M = 2.00
AISLE_CENTER_TARGET_M = AISLE_WIDTH_M / 2.0  # LiDAR/robot centre to either wall
# Lateral-centering controller: use the distance difference between both side
# boundaries when available, instead of assuming every aisle is exactly 2 m wide.
CENTERING_KP = 0.34
MAX_LATERAL_CORRECTION_M = 0.025  # no more than 2.5 cm sideways per 10 cm forward step
CENTER_DEADBAND_M = 0.035          # ignore only small measured centreline errors
MIN_CORRIDOR_WIDTH_M = ROVER_WIDTH_M + 0.20
MAX_CORRIDOR_WIDTH_M = 2.80
NOMINAL_ONE_WALL_DISTANCE_M = AISLE_WIDTH_M / 2.0
LOG_FREE = -0.40
LOG_OCC = 0.90
LOG_MIN, LOG_MAX = -5.0, 5.0


def world_to_cell(x, y):
    return int((x + HALF) / RES), int((y + HALF) / RES)


def cell_to_world(ix, iy):
    return ix * RES - HALF + RES/2, iy * RES - HALF + RES/2


def inside(ix, iy): return 0 <= ix < N and 0 <= iy < N


def identify_racks(grid):
    """Use OpenCV morphology/contours to classify rectangular racks and thin walls.

    Input is a binary occupancy grid (True = occupied). Rectangular compact
    components are rack candidates; long thin components are wall candidates.
    The classification is geometric, not semantic certainty.
    """
    occ = np.asarray(grid, dtype=np.uint8)
    if occ.shape != (N, N):
        occ = cv2.resize(occ, (N, N), interpolation=cv2.INTER_NEAREST)
    binary = (occ > 0).astype(np.uint8) * 255
    # Close tiny gaps from raster editing / imperfect SLAM edges before contouring.
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, 8)
    racks, walls = [], []
    for label in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area < 8:
            continue
        elong = max(w, h) / max(1, min(w, h))
        component = (labels[y:y+h, x:x+w] == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area = max((cv2.contourArea(c) for c in contours), default=0.0)
        rectangularity = contour_area / max(1, w*h)
        item = {
            'center': [round((x+w/2)*RES-HALF, 2), round((y+h/2)*RES-HALF, 2)],
            'bbox_m': [round(x*RES-HALF, 2), round(y*RES-HALF, 2), round(w*RES, 2), round(h*RES, 2)],
            'cells': area, 'aspect_ratio': round(elong, 2),
            'rectangularity': round(float(rectangularity), 2)
        }
        if elong >= 4.0 and min(w, h) * RES <= 0.75:
            walls.append(item)
        elif (
            area >= 100
            and elong >= 2.0
            and min(w, h) >= 8
            and max(w, h) <= 100
            and rectangularity >= 0.55
        ):
            racks.append(item)
        elif area >= 12 and w*RES <= 4.0 and h*RES <= 4.0 and rectangularity >= 0.55:
            racks.append(item)
    racks.sort(key=lambda r: (r['center'][1], r['center'][0]))
    walls.sort(key=lambda r: (r['center'][1], r['center'][0]))
    for i, rack in enumerate(racks, 1):
        rack['id'] = f'R{i:02d}'
    return racks, walls


class SLAMSimulation:
    def __init__(self, truth):
        self.truth=np.asarray(truth,dtype=bool)
        self.logodds=np.zeros((N,N),dtype=np.float32)
        self.actual=np.array([0.0,0.0,0.0],dtype=float)
        self.pose=np.array([0.0,0.0,0.0],dtype=float) # estimated x,y,yaw
        self.dock=(0.0,0.0)
        self.path=[]; self.steps=0; self.rng=np.random.default_rng(7); self.latest_scans=[]
        self.inferred_racks=np.zeros((N,N),dtype=bool)
        self.direction=(1,0)  # cardinal heading used for long straight corridor runs
        self._direction_origin=(0.0,0.0)
        self.rays=np.linspace(-math.pi,math.pi,RAY_COUNT,endpoint=False)
        self.waypoints=self._make_sweep_waypoints(); self.target_i=0
        self.mode='sweep'; self.frontier_target=None; self.frontier_count=0; self.last_frontier_refresh=-999; self.no_frontiers=False
        self.frontier_path=[]; self.frontier_path_target=None
        self._stuck_steps=0; self._last_progress_pose=self.actual.copy(); self._escape_steps=0; self._escape_sign=1
        # Diagnostic state is exposed to the UI to make centering behavior observable.
        self.centering_mode = 'waiting for side-wall observations'
        self.center_error_m = None
        self.left_wall_m = None
        self.right_wall_m = None
        self._filtered_center_error = None
        self._single_wall_side = None
        self._single_wall_streak = 0
        self._corridor_lock_steps = 0
        # A small dock footprint is reserved as known/free so the robot can start there.
        self._mark_disk(self.logodds,0,0,0.30,-0.8)

    def _make_sweep_waypoints(self):
        """Axis-aligned warehouse sweep: long straight lanes joined by 90-degree turns.

        The dock is near the centre of the editor's default map. Start at the
        dock, sweep one horizontal lane, step one lane sideways, and reverse
        direction. Rack clusters are assumed to be axis-aligned rectangles, so
        the controller avoids diagonal waypoint chasing.
        """
        pts=[]
        ys=[float(y) for y in np.arange(-9.0, 9.01, 1.0)]
        for row, y in enumerate(ys):
            xs=[float(x) for x in np.arange(-9.0, 9.01, 0.5)]
            if row % 2:
                xs.reverse()
            pts.extend((x,y) for x in xs)
        # Begin at the closest sweep point to the dock, avoiding a long initial
        # diagonal drive across an otherwise unknown map.
        pts.sort(key=lambda p: (abs(p[1]) + abs(p[0]), p[1], p[0]))
        return pts

    def _mark_disk(self, arr, x,y,r,val):
        ix,iy=world_to_cell(x,y); rad=max(1,int(r/RES))
        for yy in range(max(0,iy-rad),min(N,iy+rad+1)):
            for xx in range(max(0,ix-rad),min(N,ix+rad+1)):
                if (xx-ix)**2+(yy-iy)**2<=rad**2: arr[yy,xx]=val

    def _raycast(self, pose):
        """Simulate range-only LiDAR scans in robot-relative angles.

        Each scan item is (relative_angle, measured_range, hit_boolean).  Keeping
        the angle robot-relative is important: the old code returned world angles
        and then added yaw again during mapping, rotating scans twice.
        """
        x, y, th = pose
        result = []
        for rel in self.rays:
            ang = th + rel
            dist = LIDAR_MAX
            hit = False
            # Ray march against the ground-truth occupancy grid.
            for d in np.arange(0.10, LIDAR_MAX + 1e-9, 0.10):
                wx = x + d * math.cos(ang)
                wy = y + d * math.sin(ang)
                ix, iy = world_to_cell(wx, wy)
                if not inside(ix, iy):
                    dist = float(d)
                    break
                if self.truth[iy, ix]:
                    dist = float(d)
                    hit = True
                    break
            # A small range noise makes the simulation less idealized.
            dist = float(np.clip(dist + self.rng.normal(0.0, 0.012), 0.05, LIDAR_MAX))
            result.append((float(rel), dist, hit))
        return result

    def _update_map(self, pose, scans):
        """Inverse sensor model: clear cells along rays, occupy hit endpoints."""
        x, y, th = pose
        for rel, dist, hit in scans:
            ang = th + rel
            c, sn = math.cos(ang), math.sin(ang)
            # Leave the final 12 cm un-cleared near a hit to reduce endpoint smearing.
            free_end = max(0.10, dist - (0.12 if hit else 0.0))
            for d in np.arange(0.10, free_end, 0.12):
                wx, wy = x + d*c, y + d*sn
                ix, iy = world_to_cell(wx, wy)
                if inside(ix, iy):
                    self.logodds[iy, ix] = max(LOG_MIN, self.logodds[iy, ix] + LOG_FREE)
            if hit:
                wx, wy = x + dist*c, y + dist*sn
                ix, iy = world_to_cell(wx, wy)
                if inside(ix, iy):
                    self.logodds[iy, ix] = min(LOG_MAX, self.logodds[iy, ix] + LOG_OCC)

    def _infer_closed_rectangles(self):
        """Use OpenCV to detect closed, axis-aligned rack outlines in the partial SLAM map.

        Only accumulated SLAM occupancy is inspected; ground-truth geometry is not
        used for rack inference. Morphological closing repairs small scan gaps, and
        contour hierarchy finds enclosed interiors. Warehouse-scale contours are
        rejected so the outer boundary is not mistaken for a rack.
        """
        occ = (self.logodds > 0.55).astype(np.uint8) * 255
        if not np.any(occ):
            self.inferred_racks = np.zeros((N, N), dtype=bool)
            return
        closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        inferred = np.zeros((N, N), dtype=np.uint8)
        if hierarchy is None:
            self.inferred_racks = inferred.astype(bool)
            return
        hierarchy = hierarchy[0]
        for idx, contour in enumerate(contours):
            child = int(hierarchy[idx][2])
            if child < 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            # Typical rack blocks are rectangular and smaller than the whole room.
            if not (5 <= w <= 40 and 5 <= h <= 40):
                continue
            if w*h < 100:
                continue
            ratio = max(w, h) / max(1, min(w, h))
            if ratio > 4.0:
                continue
            contour_area = abs(cv2.contourArea(contour))
            rectangularity = contour_area / max(1, w*h)
            if rectangularity < 0.55:
                continue
            # Inspect occupancy along the perimeter and whether the inside is hollow.
            local = (closed[y:y+h, x:x+w] > 0)
            if local.shape[0] < 3 or local.shape[1] < 3:
                continue
            perimeter = np.concatenate((local[0, :], local[-1, :], local[:, 0], local[:, -1]))
            perimeter_ratio = float(perimeter.mean())
            interior = local[1:-1, 1:-1]
            interior_occ = float(interior.mean()) if interior.size else 1.0
            # Require a mostly closed outline and a substantially less occupied interior.
            if perimeter_ratio >= 0.62 and interior_occ < 0.68:
                inferred[y+1:y+h-1, x+1:x+w-1] = 1
        self.inferred_racks = inferred.astype(bool)

    def _direction_clearance(self, dx, dy):
        x,y,_=self.actual; clear=0.0
        for d in np.arange(0.15, 6.01, 0.10):
            ix,iy=world_to_cell(x+d*dx,y+d*dy)
            if not inside(ix,iy) or self.truth[iy,ix]: break
            # Also respect rack rectangles inferred from the SLAM map.
            if self.inferred_racks[iy,ix]: break
            clear=d
        return clear

    def _perpendicular_lidar_stats(self, target_angle, window=math.radians(24.0)):
        """Estimate a *longitudinal* side boundary from LiDAR hit geometry.

        Merely finding several returns near +/-90 degrees is not enough: at aisle
        mouths and rack ends, a transverse face can look like a side wall and pull
        the rover into a cross-aisle.  We fit hit endpoints in robot coordinates
        and require the surface to extend along the direction of travel (local X),
        with comparatively little lateral variation.  A valid return therefore
        represents a wall/rack side, not just an end cap.

        Returns (perpendicular distance in metres, hit count), or (None, 0) when
        the wall evidence is too sparse, angled, noisy, or transverse.
        """
        xs, ys, deltas = [], [], []
        for rel, dist, hit in self.latest_scans:
            if not hit or dist >= LIDAR_MAX * 0.97:
                continue
            delta = (rel - target_angle + math.pi) % (2 * math.pi) - math.pi
            if abs(delta) > window:
                continue
            # Endpoint coordinates in the robot's local frame.
            lx = float(dist * math.cos(rel))
            ly = float(dist * math.sin(rel))
            lateral = ly if target_angle > 0 else -ly
            if 0.12 < lateral <= 3.2:
                xs.append(lx)
                ys.append(lateral)
                deltas.append(delta)

        if len(xs) < 5 or (max(deltas) - min(deltas)) < math.radians(9.0):
            return None, 0

        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        # A side wall should generate a wide spread along the corridor and a
        # much smaller lateral spread. Reject rack end faces and corner clusters.
        longitudinal_span = float(np.percentile(x, 90) - np.percentile(x, 10))
        lateral_span = float(np.percentile(y, 90) - np.percentile(y, 10))
        if longitudinal_span < 0.24 or longitudinal_span < 1.15 * max(lateral_span, 0.015):
            return None, 0

        # Robust line fit: y ~= m*x + b. Strongly diagonal faces are not aisle walls.
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (slope * x + intercept)
        mad = float(np.median(np.abs(residual - np.median(residual))))
        if abs(float(slope)) > math.tan(math.radians(28.0)) or mad > 0.16:
            return None, 0

        # The distance at the robot's longitudinal position is the fitted
        # intercept; median is less sensitive to rasterized-map outliers.
        distance = float(np.median(y))
        if not 0.12 < distance <= 3.2:
            return None, 0
        return distance, len(xs)

    def _perpendicular_lidar_distance(self, target_angle, window=math.radians(18.0)):
        """Compatibility wrapper returning just a robust side-wall distance."""
        return self._perpendicular_lidar_stats(target_angle, window)[0]

    def _left_wall_navigation(self):
        """Nominal single-left-wall fallback; positive means shift toward the left."""
        left, _ = self._perpendicular_lidar_stats(math.pi / 2)
        if left is None or not 0.55 <= left <= 1.45:
            return None
        return float(left - NOMINAL_ONE_WALL_DISTANCE_M)

    def _right_wall_navigation(self):
        """Nominal single-right-wall fallback; positive means shift toward the left."""
        right, _ = self._perpendicular_lidar_stats(-math.pi / 2)
        if right is None or not 0.55 <= right <= 1.45:
            return None
        return float(NOMINAL_ONE_WALL_DISTANCE_M - right)

    def _lidar_corridor_center_correction(self, dx, dy):
        """Return a bounded correction toward the centre of a real aisle.

        Side distances are trusted only when LiDAR endpoints form longitudinal
        surfaces. Both sides are fused by the midpoint error. A single side is
        used only after repeated, consistent observations of a nearby wall so a
        rack end or a cross-aisle opening cannot immediately steer the rover away.
        """
        if not self.latest_scans:
            self.centering_mode = 'waiting for LiDAR scan'
            self.center_error_m = None
            self._single_wall_side = None
            self._single_wall_streak = 0
            self._corridor_lock_steps = 0
            return 0.0, 0.0

        left, left_hits = self._perpendicular_lidar_stats(math.pi / 2)
        right, right_hits = self._perpendicular_lidar_stats(-math.pi / 2)
        self.left_wall_m = round(left, 3) if left is not None else None
        self.right_wall_m = round(right, 3) if right is not None else None

        raw_error = None
        if left is not None and right is not None:
            self._single_wall_side = None
            self._single_wall_streak = 0
            estimated_width = left + right
            if MIN_CORRIDOR_WIDTH_M <= estimated_width <= MAX_CORRIDOR_WIDTH_M:
                # Positive means shift toward left; negative means shift right.
                raw_error = (left - right) / 2.0
                self._corridor_lock_steps = 8
                self.centering_mode = 'both-side aisle centering'
            else:
                self._corridor_lock_steps = 0
                self.centering_mode = 'open-space / width not trusted'
        elif left is not None or right is not None:
            side = 'left' if left is not None else 'right'
            # A single obstacle distance does not define a midpoint. In particular,
            # after a turn the outside/top warehouse wall can be the only side return;
            # steering to a presumed 1 m offset makes the rover trace that perimeter.
            # Keep reporting the observation, but never steer until both sides provide
            # reliable longitudinal surfaces again.
            self._single_wall_side = side
            self._single_wall_streak = 0
            self._corridor_lock_steps = 0
            self.centering_mode = f'{side}-only observation; midpoint unavailable'
        else:
            self._single_wall_side = None
            self._single_wall_streak = 0
            self._corridor_lock_steps = 0
            self.centering_mode = 'no reliable longitudinal side walls'

        if raw_error is None or not math.isfinite(raw_error):
            self.center_error_m = None
            self._filtered_center_error = None
            return 0.0, 0.0

        # Filter range noise without preserving stale corrections across openings.
        if self._filtered_center_error is None:
            filtered = raw_error
        else:
            filtered = 0.65 * raw_error + 0.35 * self._filtered_center_error
        self._filtered_center_error = float(filtered)
        self.center_error_m = round(float(filtered), 3)

        if abs(filtered) <= CENTER_DEADBAND_M:
            return 0.0, 0.0

        lateral_step = float(np.clip(filtered * CENTERING_KP,
                                     -MAX_LATERAL_CORRECTION_M,
                                     MAX_LATERAL_CORRECTION_M))
        # Correction is returned in world coordinates using the actual body
        # heading's left-normal. This avoids a sign/frame mismatch during turns.
        heading = float(self.actual[2])
        px, py = -math.sin(heading), math.cos(heading)
        return lateral_step * px, lateral_step * py

    def _choose_cardinal_direction(self):
        """Use OpenCV's current partial map plus LiDAR rays to choose an aisle.

        First reject directions blocked by known occupied map cells. Then score
        remaining cardinal directions by (1) LiDAR rays that reach far without a
        hit, (2) unknown cells ahead that can be explored, and (3) open map run.
        Ties prefer a perpendicular turn to the previous heading; random choice
        breaks a remaining tie. Ground truth is deliberately not used for scoring.
        """
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        occupied = ((self.logodds > 0.65) | self.inferred_racks).astype(np.uint8) * 255
        # OpenCV closes tiny speckles/gaps so a noisy map edge doesn't create a
        # false passage. Keep the kernel small to avoid closing real narrow aisles.
        cv_occ = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        x, y, _ = self.actual
        # Scan angles are relative to the simulator body pose, not the SLAM estimate.
        heading = float(self.actual[2])
        scan_scores = {}
        for dx, dy in dirs:
            world_ang = math.atan2(dy, dx)
            vals = []
            for rel, dist, hit in self.latest_scans:
                ang = (heading + rel - world_ang + math.pi) % (2*math.pi) - math.pi
                if abs(ang) <= math.radians(18):
                    vals.append((float(dist), bool(hit)))
            if vals:
                # Favor directions with many rays reaching the sensor limit and few hits.
                free_fraction = sum((not hit) or dist >= LIDAR_MAX*0.92 for dist, hit in vals) / len(vals)
                reach = float(np.mean([min(dist, LIDAR_MAX) for dist, _ in vals])) / LIDAR_MAX
                scan_scores[(dx, dy)] = 0.65*free_fraction + 0.35*reach
            else:
                scan_scores[(dx, dy)] = 0.0

        scores = {}
        clearances = {}
        unknown_scores = {}
        for dx, dy in dirs:
            clearance = 0.0
            unknown = []
            for dist in np.arange(0.15, LIDAR_MAX + 0.01, 0.10):
                ix, iy = world_to_cell(x + dist*dx, y + dist*dy)
                if not inside(ix, iy) or cv_occ[iy, ix]:
                    break
                clearance = float(dist)
                unknown.append(abs(float(self.logodds[iy, ix])) <= 0.15)
            clearances[(dx, dy)] = clearance
            unknown_scores[(dx, dy)] = float(np.mean(unknown)) if unknown else 0.0
            # Scan openness is intentionally weighted strongly, but unknown-space
            # gain also matters so the rover does not endlessly follow known lanes.
            scores[(dx, dy)] = (0.55*clearance + 2.2*scan_scores[(dx, dy)] +
                                1.6*unknown_scores[(dx, dy)])

        viable = [d for d in dirs if clearances[d] >= 0.45 and scan_scores[d] >= 0.15]
        if not viable:
            # If scans are sparse/ambiguous, fall back to map-visible free runs.
            viable = [d for d in dirs if clearances[d] >= 0.45]
        if not viable:
            return self.direction
        best = max(scores[d] for d in viable)
        candidates = [d for d in viable if scores[d] >= best - 0.18]
        old = self.direction
        perpendicular = [d for d in candidates if d[0]*old[0] + d[1]*old[1] == 0]
        if perpendicular:
            return perpendicular[int(self.rng.integers(len(perpendicular)))]
        if old in candidates:
            return old
        return candidates[int(self.rng.integers(len(candidates)))]

    def _scan_match(self, predicted, scans):
        """Conservative local scan matching against accumulated occupied cells.

        In a long straight aisle, translation along the aisle is weakly observable.
        The previous matcher searched +/-12 cm each step and could repeatedly pick
        arbitrary offsets along a featureless wall, creating more drift than it fixed.
        This version searches a smaller neighbourhood, penalizes unnecessary motion,
        and accepts a correction only when it improves the scan score meaningfully.
        """
        if self.steps < 15 or np.count_nonzero(self.logodds > 1.0) < 20:
            return predicted
        hits = [(d * math.cos(rel), d * math.sin(rel))
                for rel, d, hit in scans if hit and d < 4.5]
        if len(hits) > 55:
            hits = hits[::max(1, len(hits) // 55)]
        if len(hits) < 8:
            return predicted

        def scan_score(dx, dy, da):
            x = predicted[0] + dx
            y = predicted[1] + dy
            th = predicted[2] + da
            c, sn = math.cos(th), math.sin(th)
            vals = []
            for lx, ly in hits:
                wx = x + lx * c - ly * sn
                wy = y + lx * sn + ly * c
                ix, iy = world_to_cell(wx, wy)
                if not inside(ix, iy):
                    continue
                patch = self.logodds[max(0, iy-1):min(N, iy+2),
                                     max(0, ix-1):min(N, ix+2)]
                vals.append(float(np.max(patch)))
            if not vals:
                return -1e9
            alignment = float(np.mean(vals))
            # Keep the estimate close to odometry unless the map gives clear evidence.
            prior_penalty = 2.2 * (abs(dx) + abs(dy)) + 0.55 * abs(da)
            return alignment - prior_penalty

        best = predicted.copy()
        baseline = scan_score(0.0, 0.0, 0.0)
        bestscore = baseline
        offsets_xy = (-0.06, -0.03, 0.0, 0.03, 0.06)
        offsets_yaw = (-0.03, 0.0, 0.03)
        min_gain = 0.025
        for dx in offsets_xy:
            for dy in offsets_xy:
                for da in offsets_yaw:
                    if dx == 0.0 and dy == 0.0 and da == 0.0:
                        continue
                    score = scan_score(dx, dy, da)
                    if score > baseline + min_gain and score > bestscore:
                        bestscore = score
                        best = np.array([predicted[0] + dx,
                                         predicted[1] + dy,
                                         predicted[2] + da])
        return best

    def _collision(self,x,y):
        # 25 cm radius rover; collision checks use the simulated ground truth only
        for a in np.linspace(0,2*math.pi,12,endpoint=False):
            ix,iy=world_to_cell(x+0.25*math.cos(a),y+0.25*math.sin(a))
            if not inside(ix,iy) or self.truth[iy,ix] or self.inferred_racks[iy,ix]: return True
        return False

    def _choose_frontier(self):
        """Choose a reachable-looking frontier: known free next to unknown cells.

        Candidates are sampled on a coarse grid to keep the pure-Python simulator
        responsive. Ground truth is deliberately not consulted by this planner.
        """
        known_free = self.logodds < -0.15
        unknown = np.abs(self.logodds) <= 0.15
        adjacent_unknown = np.zeros_like(unknown)
        adjacent_unknown[1:] |= unknown[:-1]
        adjacent_unknown[:-1] |= unknown[1:]
        adjacent_unknown[:, 1:] |= unknown[:, :-1]
        adjacent_unknown[:, :-1] |= unknown[:, 1:]
        frontier = known_free & adjacent_unknown
        # Keep candidates away from the immediate rover footprint and score by
        # travel distance with a small preference for frontier-rich neighbourhoods.
        ax, ay, _ = self.actual
        candidates=[]
        for iy in range(3, N-3, 5):
            for ix in range(3, N-3, 5):
                if not frontier[iy, ix]:
                    continue
                x, y = cell_to_world(ix, iy)
                dist = math.hypot(x-ax, y-ay)
                if dist < 0.65 or dist > 10.0:
                    continue
                density = int(frontier[max(0,iy-5):min(N,iy+6), max(0,ix-5):min(N,ix+6)].sum())
                score = dist - min(density, 40) * 0.012
                candidates.append((score, x, y))
        if not candidates:
            # Fallback: use OpenCV to find sizeable unknown regions and drive to
            # a reachable free cell on the edge of the most promising region.
            # This uses only the partial SLAM map, never the simulator ground truth.
            target = self._opencv_guess_unexplored_target(known_free, unknown)
            self.frontier_count = 0
            return target
        candidates.sort(key=lambda c:c[0])
        self.frontier_count=len(candidates)
        return (candidates[0][1], candidates[0][2])

    def _opencv_guess_unexplored_target(self, known_free=None, unknown=None):
        """Guess where to explore next when ordinary frontier sampling fails.

        OpenCV groups unknown cells into regions, filters tiny noise, and ranks
        their reachable boundaries. The goal is a known-free cell next to a
        likely unexplored region, not the unknown region's interior.
        """
        if known_free is None:
            known_free = self.logodds < -0.15
        if unknown is None:
            unknown = np.abs(self.logodds) <= 0.15
        if not np.any(known_free) or not np.any(unknown):
            return None

        # Close pinholes and remove isolated unknown pixels so the regions are
        # coherent enough for contour/connected-component analysis.
        unknown_u8 = unknown.astype(np.uint8) * 255
        unknown_u8 = cv2.morphologyEx(unknown_u8, cv2.MORPH_CLOSE,
                                      np.ones((3, 3), np.uint8), iterations=1)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(unknown_u8, 8)
        # A frontier candidate must be known free and immediately adjacent to
        # one of these unknown components.
        adjacent = cv2.dilate(unknown_u8, np.ones((3, 3), np.uint8), iterations=1) > 0
        reachable_edge = known_free & adjacent
        if not np.any(reachable_edge):
            # Slightly widen the search for imperfectly connected map edges.
            adjacent = cv2.dilate(unknown_u8, np.ones((5, 5), np.uint8), iterations=1) > 0
            reachable_edge = known_free & adjacent
        if not np.any(reachable_edge):
            return None

        ax, ay, _ = self.actual
        ranked = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 18:
                continue
            region = labels == label
            edge = reachable_edge & cv2.dilate(region.astype(np.uint8),
                                                np.ones((3, 3), np.uint8),
                                                iterations=1).astype(bool)
            ys, xs = np.where(edge)
            if len(xs) == 0:
                continue
            # Subsample very long boundaries, retaining a broad set of goals.
            if len(xs) > 500:
                take = np.linspace(0, len(xs)-1, 500).astype(int)
                xs, ys = xs[take], ys[take]
            wx = xs * RES - HALF + RES / 2.0
            wy = ys * RES - HALF + RES / 2.0
            dists = np.hypot(wx - ax, wy - ay)
            valid = (dists >= 0.65) & (dists <= 12.0)
            if not np.any(valid):
                continue
            indices = np.where(valid)[0]
            # Prefer a nearby accessible edge, with a modest bonus for larger
            # unexplored regions. This prevents distant map-border unknowns
            # from dominating every fallback decision.
            best_idx = indices[np.argmin(dists[indices])]
            distance = float(dists[best_idx])
            area_bonus = min(area, 2500) ** 0.5 * 0.012
            score = distance - area_bonus
            ranked.append((score, float(wx[best_idx]), float(wy[best_idx])))

        if not ranked:
            return None
        ranked.sort(key=lambda item: item[0])
        return (ranked[0][1], ranked[0][2])

    def _plan_map_path(self, target):
        """A* path to a frontier using the partial SLAM map, not ground truth.

        Known obstacles are inflated by approximately the rover radius. Known-free
        cells are cheaper than unknown cells, so the planner prefers mapped aisles
        but can cautiously approach a frontier through unmapped space when needed.
        The per-step collision guard still rejects a move that the map got wrong.
        """
        if target is None:
            return []
        sx, sy = world_to_cell(float(self.actual[0]), float(self.actual[1]))
        gx, gy = world_to_cell(float(target[0]), float(target[1]))
        if not inside(sx, sy) or not inside(gx, gy):
            return []

        occupied = self.known_occupied().astype(np.uint8) * 255
        inflated = cv2.dilate(occupied, np.ones((5, 5), np.uint8), iterations=1) > 0
        inflated[sy, sx] = False
        inflated[gy, gx] = False
        known_free = self.logodds < -0.15

        start, goal = (sx, sy), (gx, gy)
        frontier = [(abs(gx-sx) + abs(gy-sy), 0.0, start)]
        came_from = {}
        cost_so_far = {start: 0.0}
        expanded = 0
        max_expanded = 5000
        neighbours = ((1, 0), (-1, 0), (0, 1), (0, -1))

        while frontier and expanded < max_expanded:
            _, current_cost, current = heapq.heappop(frontier)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            if current_cost > cost_so_far.get(current, math.inf) + 1e-9:
                continue
            expanded += 1
            cx, cy = current
            for ox, oy in neighbours:
                nx, ny = cx + ox, cy + oy
                if not inside(nx, ny) or inflated[ny, nx]:
                    continue
                step_cost = 1.0 if known_free[ny, nx] else 2.4
                new_cost = current_cost + step_cost
                node = (nx, ny)
                if new_cost + 1e-9 < cost_so_far.get(node, math.inf):
                    cost_so_far[node] = new_cost
                    came_from[node] = current
                    heuristic = abs(gx-nx) + abs(gy-ny)
                    heapq.heappush(frontier, (new_cost + heuristic, new_cost, node))
        return []

    def _frontier_navigation_direction(self):
        """Choose the next cardinal step along a map-planned frontier route."""
        if self.coverage_estimate() < 0.045:
            return None

        tx, ty = self.frontier_target if self.frontier_target is not None else (None, None)
        target_distance = (math.hypot(tx - float(self.actual[0]), ty - float(self.actual[1]))
                           if tx is not None else math.inf)
        since_plan = self.steps - self.last_frontier_refresh
        needs_plan = (since_plan >= 45 or target_distance < 0.25 or
                      (not self.frontier_path and since_plan >= 6) or
                      (self.frontier_path_target is None and since_plan >= 6))
        if needs_plan:
            target = self._choose_frontier()
            self.last_frontier_refresh = self.steps
            self.frontier_target = target
            self.frontier_path = self._plan_map_path(target) if target is not None else []
            self.frontier_path_target = target if self.frontier_path else None

        if len(self.frontier_path) < 2:
            return None

        ix, iy = world_to_cell(float(self.actual[0]), float(self.actual[1]))
        index = min(range(len(self.frontier_path)),
                    key=lambda i: abs(self.frontier_path[i][0]-ix) + abs(self.frontier_path[i][1]-iy))
        if index >= len(self.frontier_path) - 1:
            self.frontier_path = []
            self.frontier_path_target = None
            return None
        here = self.frontier_path[index]
        nxt = self.frontier_path[index + 1]
        dx, dy = nxt[0] - here[0], nxt[1] - here[1]
        if abs(dx) + abs(dy) != 1:
            self.frontier_path = []
            self.frontier_path_target = None
            return None
        return (int(dx), int(dy))

    @staticmethod
    def _axis_waypoint(x, y, tx, ty, yaw):
        """Choose a cardinal leg (no diagonal motion) toward a target."""
        dx, dy = tx-x, ty-y
        # Prefer the axis with the larger remaining error; ties preserve heading.
        if abs(dx) > abs(dy) + 0.05:
            return (1.0 if dx >= 0 else -1.0), 0.0
        if abs(dy) > abs(dx) + 0.05:
            return 0.0, (1.0 if dy >= 0 else -1.0)
        hx, hy = math.cos(yaw), math.sin(yaw)
        return (1.0 if hx >= 0 else -1.0, 0.0) if abs(hx) >= abs(hy) else (0.0, 1.0 if hy >= 0 else -1.0)

    def step(self):
        """Advance one simulation step with obstacle-triggered turns and steering.

        Exploration remains cardinal at the decision level, but when side LiDAR
        detects an aisle the rover uses a bounded heading change while moving
        forward. This replaces the old instantaneous sideways position shift, which
        was not consistent with differential-drive motion.
        """
        ax, ay, ath = [float(v) for v in self.actual]
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        if self.direction not in dirs:
            self.direction = (1, 0)
        planned_direction = self._frontier_navigation_direction()
        if planned_direction in dirs:
            self.direction = planned_direction
        dx, dy = self.direction
        step_len = 0.10
        nominal_yaw = math.atan2(dy, dx)
        straight_x, straight_y = ax + step_len * dx, ay + step_len * dy
        blocked_ahead = self._collision(straight_x, straight_y)
        turn_requested = False

        if blocked_ahead:
            # A real collision check caught an obstacle missing from the partial map.
            # Discard this route and rebuild it after the new scan is integrated.
            self.frontier_path = []
            self.frontier_path_target = None
            # Never choose a direction whose first movement would collide.
            safe = []
            for d in dirs:
                tx, ty = ax + step_len * d[0], ay + step_len * d[1]
                if not self._collision(tx, ty) and self._direction_clearance(*d) >= 0.35:
                    safe.append(d)
            if safe:
                old = self.direction

                # Refresh the SLAM-frontier goal periodically. The previous
                # controller always preferred any non-reverse turn, so at a lane
                # end it often turned along the warehouse perimeter and kept
                # orbiting the outer wall instead of returning through the aisle.
                current_target_dist = (math.hypot(self.frontier_target[0] - ax,
                                                   self.frontier_target[1] - ay)
                                       if self.frontier_target is not None else math.inf)
                if (self.frontier_target is None or current_target_dist < 0.65 or
                        self.steps - self.last_frontier_refresh >= 18):
                    self.frontier_target = self._choose_frontier()
                    self.last_frontier_refresh = self.steps

                chosen = None
                if self.frontier_target is not None:
                    tx, ty = self.frontier_target
                    before = math.hypot(tx - ax, ty - ay)
                    # Rank safe turns by progress toward the map-derived frontier.
                    # A reverse step is allowed when it is the best way back into
                    # an aisle; this is the key fix for perimeter-following loops.
                    progress = {}
                    for d in safe:
                        after = math.hypot(tx - (ax + step_len*d[0]),
                                           ty - (ay + step_len*d[1]))
                        progress[d] = before - after
                    best_progress = max(progress.values())
                    target_choices = [d for d in safe if progress[d] >= best_progress - 0.012]
                    suggested = self._choose_cardinal_direction()
                    if suggested in target_choices:
                        chosen = suggested
                    else:
                        # Prefer a well-observed clear run only among equally
                        # useful target-directed turns.
                        def local_rank(d):
                            unknown = 0
                            clear = 0
                            for dist in np.arange(0.2, min(LIDAR_MAX, 3.0) + 0.01, 0.2):
                                ix, iy = world_to_cell(ax + dist*d[0], ay + dist*d[1])
                                if not inside(ix, iy) or self.inferred_racks[iy, ix] or self.logodds[iy, ix] > 0.65:
                                    break
                                clear += 1
                                if abs(float(self.logodds[iy, ix])) <= 0.15:
                                    unknown += 1
                            return (clear, unknown, int(d[0]*old[0] + d[1]*old[1] == 0))
                        best_local = max(local_rank(d) for d in target_choices)
                        tied = [d for d in target_choices if local_rank(d) == best_local]
                        chosen = tied[int(self.rng.integers(len(tied)))]

                if chosen is None:
                    # With no reachable frontier goal, use map + LiDAR open-run
                    # scoring. Avoid reverse only in this fallback case.
                    pool = [d for d in safe if d != (-old[0], -old[1])] or safe
                    suggested = self._choose_cardinal_direction()
                    if suggested in pool:
                        chosen = suggested
                    else:
                        def direction_rank(d):
                            unknown = 0
                            clear = 0
                            for dist in np.arange(0.2, min(LIDAR_MAX, 3.0) + 0.01, 0.2):
                                ix, iy = world_to_cell(ax + dist*d[0], ay + dist*d[1])
                                if not inside(ix, iy) or self.inferred_racks[iy, ix] or self.logodds[iy, ix] > 0.65:
                                    break
                                clear += 1
                                if abs(float(self.logodds[iy, ix])) <= 0.15:
                                    unknown += 1
                            perpendicular = int(d[0]*old[0] + d[1]*old[1] == 0)
                            return (clear, unknown, perpendicular)
                        ranks = {d: direction_rank(d) for d in pool}
                        best_rank = max(ranks.values())
                        tied = [d for d in pool if ranks[d] == best_rank]
                        if len(tied) > 1:
                            perpendicular = [d for d in tied if d[0]*old[0] + d[1]*old[1] == 0]
                            if perpendicular:
                                tied = perpendicular
                        chosen = tied[int(self.rng.integers(len(tied)))]

                self.direction = chosen
                dx, dy = self.direction
            else:
                # Fully boxed in: turn around, then re-scan without moving.
                self.direction = (-dx, -dy)
                dx, dy = self.direction
            turn_requested = True
            nominal_yaw = math.atan2(dy, dx)

        heading_error_to_lane = (nominal_yaw - ath + math.pi) % (2 * math.pi) - math.pi
        if turn_requested or abs(heading_error_to_lane) > 0.22:
            # A turn/junction invalidates the previous aisle geometry. Require fresh
            # simultaneous left+right wall observations before centering resumes.
            self._corridor_lock_steps = 0
            self._single_wall_side = None
            self._single_wall_streak = 0
            self._filtered_center_error = None
            # Rotate in place until the rover is broadly aligned with the chosen lane.
            nth = ath + float(np.clip(heading_error_to_lane, -0.30, 0.30))
            nx, ny = ax, ay
        else:
            # Use the side-distance difference as a desired lateral displacement,
            # then convert it to a steering angle over the forward step length.
            cx, cy = self._lidar_corridor_center_correction(dx, dy)
            lateral_cmd = cx * (-dy) + cy * dx
            steering_offset = math.atan2(lateral_cmd, step_len)
            desired_yaw = nominal_yaw + steering_offset
            yaw_error = (desired_yaw - ath + math.pi) % (2 * math.pi) - math.pi
            # Limit yaw rate, so the path curves smoothly instead of snapping sideways.
            nth = ath + float(np.clip(yaw_error, -0.12, 0.12))
            nx = ax + step_len * math.cos(nth)
            ny = ay + step_len * math.sin(nth)
            if self._collision(nx, ny):
                # Prefer a safe straight step over a steering command that clips a rack.
                nth = nominal_yaw
                nx, ny = straight_x, straight_y
                if self._collision(nx, ny):
                    nx, ny = ax, ay
                    turn_error = (nominal_yaw - ath + math.pi) % (2 * math.pi) - math.pi
                    nth = ath + float(np.clip(turn_error, -0.30, 0.30))

        self.actual = np.array([nx, ny, nth], dtype=float)

        # Noisy odometry prediction and map update. The noise is smaller than in the
        # first version (which used 1.2 cm per 10 cm step and visibly accumulated drift).
        odx = nx - ax + float(self.rng.normal(0, 0.003))
        ody = ny - ay + float(self.rng.normal(0, 0.003))
        odth = nth - ath + float(self.rng.normal(0, 0.004))
        pred = self.pose.copy()
        pred[0] += odx
        pred[1] += ody
        pred[2] += odth
        scans = self._raycast(self.actual)
        self.latest_scans = scans
        self.pose = self._scan_match(pred, scans)
        self._update_map(self.pose, scans)
        if self.steps % 12 == 0:
            self._infer_closed_rectangles()
        self.path.append((float(self.pose[0]), float(self.pose[1])))
        self.steps += 1
        self.mode = 'frontier' if self.coverage_estimate() >= 0.12 else 'sweep'
        self.frontier_count = int(np.count_nonzero(np.abs(self.logodds) <= 0.15))
        self.no_frontiers = self.frontier_count == 0

    def _clearance(self, angle):
        x,y,_=self.actual; clear=0.0
        for d in np.arange(0.15,1.1,0.1):
            ix,iy=world_to_cell(x+d*math.cos(angle),y+d*math.sin(angle))
            if not inside(ix,iy) or self.truth[iy,ix]: break
            clear=d
        return clear

    def known_mask(self): return np.abs(self.logodds)>0.15
    def known_occupied(self): return (self.logodds>0.45) | self.inferred_racks
    def coverage_estimate(self): return float(np.mean(self.known_mask()))

    def return_to_dock(self):
        # In the simulation return directly toward dock with obstacle avoidance.
        for _ in range(1200):
            x,y,th=self.actual; dist=math.hypot(x,y)
            if dist<0.25: break
            desired=math.atan2(-y,-x); err=(desired-th+math.pi)%(2*math.pi)-math.pi
            if abs(err)>0.45:
                nth=th+float(np.clip(err,-0.2,0.2)); nx,ny=x,y
            else:
                nth=th+float(np.clip(err,-0.1,0.1)); nx=x+0.12*math.cos(nth); ny=y+0.12*math.sin(nth)
                if self._collision(nx,ny): nth=th+0.3; nx,ny=x,y
            self.actual=np.array([nx,ny,nth])
            self.pose[0]+=nx-x+float(self.rng.normal(0,0.003)); self.pose[1]+=ny-y+float(self.rng.normal(0,0.003)); self.pose[2]+=nth-th
            scans=self._raycast(self.actual); self.latest_scans=scans; self.pose=self._scan_match(self.pose,scans); self._update_map(self.pose,scans)
            self.path.append((float(self.pose[0]),float(self.pose[1]))); self.steps+=1

    def summary(self):
        return {'steps':self.steps,'coverage_percent':round(100*self.coverage_estimate(),1),'mode':self.mode,'frontier_candidates':self.frontier_count,'estimated_pose':[round(float(v),2) for v in self.pose], 'true_pose':[round(float(v),2) for v in self.actual], 'centering_mode':self.centering_mode, 'center_error_m':self.center_error_m, 'left_wall_m':self.left_wall_m, 'right_wall_m':self.right_wall_m}
