"""Regression tests for the LiDAR aisle-centering controller.

Run with: python -m unittest discover -s tests -v
Requires numpy and opencv-python-headless (already in requirements.txt).
"""
import math
import unittest

import numpy as np

from slam import N, SLAMSimulation, world_to_cell


def make_side_scan(left=None, right=None):
    """Create a deterministic synthetic 120-beam scan with parallel side walls."""
    result = []
    for rel in np.linspace(-math.pi, math.pi, 120, endpoint=False):
        hit, dist = False, 6.0
        if left is not None:
            delta = (rel - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
            if abs(delta) <= math.radians(18):
                dist, hit = left / max(0.01, math.cos(delta)), True
        if right is not None:
            delta = (rel + math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
            if abs(delta) <= math.radians(18):
                dist, hit = right / max(0.01, math.cos(delta)), True
        result.append((float(rel), float(dist), hit))
    return result


class AisleCenteringTests(unittest.TestCase):
    def setUp(self):
        self.empty_truth = np.zeros((N, N), dtype=bool)

    def test_asymmetric_walls_produce_correction_toward_midpoint(self):
        sim = SLAMSimulation(self.empty_truth)
        sim.latest_scans = make_side_scan(left=1.1, right=1.7)
        _, lateral = sim._lidar_corridor_center_correction(1, 0)
        self.assertLess(lateral, 0.0, "robot should steer right toward the aisle midpoint")
        self.assertAlmostEqual(sim.center_error_m, -0.3, places=2)

    def test_opposite_offset_produces_opposite_correction(self):
        sim = SLAMSimulation(self.empty_truth)
        sim.latest_scans = make_side_scan(left=1.4, right=1.0)
        _, lateral = sim._lidar_corridor_center_correction(1, 0)
        self.assertGreater(lateral, 0.0, "robot should steer left toward the aisle midpoint")

    def test_single_wall_never_assumes_a_midpoint(self):
        # A single wall, even after a previously observed aisle, cannot tell the
        # rover where the opposite boundary is. This prevents perimeter tracking.
        sim = SLAMSimulation(self.empty_truth)
        sim.latest_scans = make_side_scan(left=1.2, right=1.2)
        sim._lidar_corridor_center_correction(1, 0)
        sim.latest_scans = make_side_scan(left=1.2, right=None)
        for _ in range(4):
            self.assertEqual(sim._lidar_corridor_center_correction(1, 0), (0.0, 0.0))
        self.assertIn('midpoint unavailable', sim.centering_mode)

        sim.latest_scans = make_side_scan(left=None, right=1.2)
        self.assertEqual(sim._lidar_corridor_center_correction(1, 0), (0.0, 0.0))
        self.assertIn('midpoint unavailable', sim.centering_mode)

    def test_centered_or_open_space_does_not_create_false_correction(self):
        centered = SLAMSimulation(self.empty_truth)
        centered.latest_scans = make_side_scan(left=1.3, right=1.3)
        self.assertEqual(centered._lidar_corridor_center_correction(1, 0), (0.0, 0.0))

        open_space = SLAMSimulation(self.empty_truth)
        open_space.latest_scans = make_side_scan(left=4.8, right=4.8)
        self.assertEqual(open_space._lidar_corridor_center_correction(1, 0), (0.0, 0.0))


    def test_transverse_rack_end_is_not_mistaken_for_a_side_wall(self):
        sim = SLAMSimulation(self.empty_truth)
        # A flat rack-end face at local x=0.25 m generates side-sector hits whose
        # endpoints vary laterally but have almost no longitudinal spread.
        scan = []
        for rel in np.linspace(-math.pi, math.pi, 120, endpoint=False):
            hit, dist = False, 6.0
            if math.radians(70) <= rel <= math.radians(85):
                c = math.cos(rel)
                if c > 0:
                    dist, hit = 0.25 / c, True
            scan.append((float(rel), float(dist), hit))
        sim.latest_scans = scan
        distance, hits = sim._perpendicular_lidar_stats(math.pi / 2)
        self.assertIsNone(distance)
        self.assertEqual(hits, 0)

    def test_frontier_navigation_keeps_exploration_off_the_outer_perimeter(self):
        # A warehouse with horizontal rack rows and cross-aisle gaps. The previous
        # turn-only explorer ran to the right boundary, then followed the outer
        # perimeter because it never followed its computed frontier goals.
        truth = np.zeros((N, N), dtype=bool)
        truth[8:12, :] = True
        truth[-12:-8, :] = True
        truth[:, 8:12] = True
        truth[:, -12:-8] = True
        truth[8:12, 142:158] = False  # dock opening
        for row in (42, 82, 122, 162, 202, 242):
            truth[row:row+12, 38:112] = True
            truth[row:row+12, 142:216] = True
            truth[row:row+12, 246:278] = True

        sim = SLAMSimulation(truth)
        for _ in range(150):
            sim.step()
            self.assertFalse(sim._collision(*sim.actual[:2]))

        actual_x = float(sim.actual[0])
        self.assertLess(abs(actual_x), 12.0,
                        "frontier routing should not simply follow the outer side wall")
        path = np.asarray(sim.path, dtype=float)
        self.assertGreater(float(np.ptp(path[:, 1])), 1.5,
                           "rover should explore more than one horizontal lane")
        self.assertIsNotNone(sim.frontier_target)

    def test_rover_steers_to_center_in_real_simulated_aisle(self):
        truth = np.zeros((N, N), dtype=bool)
        # Parallel side boundaries form a 2 m clear aisle from x=-8 to x=8.
        truth[138:140, 70:230] = True  # lower wall, approximately y=-1.2 to -1.0 m
        truth[160:162, 70:230] = True  # upper wall, approximately y=+1.0 to +1.2 m
        sim = SLAMSimulation(truth)
        sim.actual = np.array([0.0, 0.35, 0.0])
        sim.pose = np.array([0.0, 0.35, 0.0])
        sim.direction = (1, 0)
        previous = sim.actual.copy()

        for _ in range(65):
            sim.step()
            movement = float(np.linalg.norm(sim.actual[:2] - previous[:2]))
            # Turns may rotate in place; while moving, keep each step bounded.
            self.assertLessEqual(movement, 0.101)
            self.assertFalse(sim._collision(*sim.actual[:2]), "rover footprint should not collide")
            previous = sim.actual.copy()

        self.assertLess(abs(float(sim.actual[1])), 0.12)
        self.assertLess(abs(float(sim.pose[1])), 0.20)
        self.assertLess(abs(float(sim.pose[0] - sim.actual[0])), 0.65)
        self.assertEqual(sim.centering_mode, 'both-side aisle centering')


if __name__ == '__main__':
    unittest.main()

class OpenCVUnexploredFallbackTests(unittest.TestCase):
    def test_opencv_fallback_returns_reachable_free_cell_next_to_unknown_region(self):
        truth = np.zeros((N, N), dtype=bool)
        sim = SLAMSimulation(truth)
        sim.logodds[:, :] = 0.0  # unknown map
        sim.logodds[140:161, 140:176] = -0.8  # known-free region around rover
        known_free = sim.logodds < -0.15
        unknown = np.abs(sim.logodds) <= 0.15

        target = sim._opencv_guess_unexplored_target(known_free, unknown)
        self.assertIsNotNone(target)
        tx, ty = world_to_cell(*target)
        self.assertTrue(known_free[ty, tx])
        neighbourhood = unknown[max(0, ty-2):min(N, ty+3), max(0, tx-2):min(N, tx+3)]
        self.assertTrue(np.any(neighbourhood))

    def test_frontier_picker_uses_opencv_fallback_when_sampled_frontiers_are_missing(self):
        truth = np.zeros((N, N), dtype=bool)
        sim = SLAMSimulation(truth)
        sim.logodds[:, :] = 0.0
        sim.logodds[140:161, 140:176] = -0.8
        # A small reachable boundary may be missed by the normal coarse-grid sampler.
        target = sim._choose_frontier()
        self.assertIsNotNone(target)
        tx, ty = world_to_cell(*target)
        self.assertLess(float(sim.logodds[ty, tx]), -0.15)
