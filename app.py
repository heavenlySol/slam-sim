from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
from threading import Thread, Lock
import json, time, io, base64, re, math, heapq
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from slam import SLAMSimulation, identify_racks

BASE = Path(__file__).resolve().parent
MAP_DIR = BASE / 'data' / 'maps'
OUT_DIR = BASE / 'data' / 'runs'
MAP_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
app = Flask(__name__)

@app.errorhandler(Exception)
def api_error_response(exc):
    # The frontend expects JSON from /api/*; Flask's default HTML error page
    # otherwise appears as the confusing browser error: JSON.parse unexpected character.
    from werkzeug.exceptions import HTTPException
    if isinstance(exc, HTTPException):
        status = exc.code or 500
        message = exc.description
    else:
        app.logger.exception('Unhandled request error', exc_info=exc)
        status = 500
        message = str(exc) or 'Internal server error'
    if request.path.startswith('/api/'):
        return jsonify(error=message), status
    if isinstance(exc, HTTPException):
        return exc
    return 'Internal server error', 500

STATE_LOCK = Lock()
RUNS = {}
GRID_N = 300


def safe_name(name):
    name = re.sub(r'[^a-zA-Z0-9_-]+', '_', (name or 'warehouse').strip())[:48]
    return name or 'warehouse'


def load_map(name):
    p = MAP_DIR / f'{safe_name(name)}.json'
    if not p.exists(): return None
    data = json.loads(p.read_text())
    rows = data.get('rows', [])
    if len(rows) != GRID_N or any(len(r) != GRID_N for r in rows): return None
    return np.array([[c == '1' for c in row] for row in rows], dtype=bool)


def save_map(name, grid):
    rows = [''.join('1' if v else '0' for v in row) for row in grid]
    p = MAP_DIR / f'{safe_name(name)}.json'
    p.write_text(json.dumps({'name': safe_name(name), 'resolution_m': 0.1, 'size_m': 30, 'rows': rows}))
    return p



def slam_live_png(sim, size=600):
    """Render current occupancy, estimated trajectory, LiDAR rays, and rover."""
    occ = sim.known_occupied()
    known = sim.known_mask()
    rgb = np.full((GRID_N, GRID_N, 3), (218, 220, 223), dtype=np.uint8)
    rgb[known] = (250, 250, 250)
    rgb[occ] = (48, 53, 61)
    img = Image.fromarray(np.flipud(rgb)).resize((size, size), Image.Resampling.NEAREST)
    d = ImageDraw.Draw(img)
    def xy(x, y):
        return ((x + 15.0) / 30.0 * size, (15.0 - y) / 30.0 * size)
    # Estimated trajectory (red), updated while the simulation is running.
    if len(sim.path) > 1:
        d.line([xy(x, y) for x, y in sim.path], fill=(235, 35, 35), width=2)
    # Current scan rays (green); use estimated pose and robot-relative angles.
    rx, ry, yaw = [float(v) for v in sim.pose]
    for rel, dist, hit in sim.latest_scans:
        a = yaw + rel
        ex, ey = rx + dist * math.cos(a), ry + dist * math.sin(a)
        x0, y0 = xy(rx, ry); x1, y1 = xy(ex, ey)
        d.line((x0, y0, x1, y1), fill=(0, 170, 80), width=1)
        if hit:
            d.ellipse((x1-2, y1-2, x1+2, y1+2), fill=(0, 190, 90))
    # Dock marker and current estimated rover position.
    dx, dy = xy(0.0, 0.0)
    d.ellipse((dx-4, dy-4, dx+4, dy+4), fill=(255, 165, 0), outline=(0, 0, 0))
    sx, sy = xy(rx, ry)
    d.ellipse((sx-6, sy-6, sx+6, sy+6), fill=(20, 90, 255), outline=(0, 0, 100), width=2)
    # Heading indicator makes turning visible.
    hx, hy = xy(rx + 0.55*math.cos(yaw), ry + 0.55*math.sin(yaw))
    d.line((sx, sy, hx, hy), fill=(20, 90, 255), width=3)
    return img

def grid_png(grid, racks=None, path=None, pose=None, title='Warehouse map', visit_order=None):
    # grid rows represent increasing world Y; image flips vertically.
    rgb = np.zeros((GRID_N, GRID_N, 3), dtype=np.uint8)
    rgb[:] = (250, 250, 250)
    rgb[grid] = (48, 53, 61)
    img = Image.fromarray(np.flipud(rgb)).resize((900, 900), Image.Resampling.NEAREST)
    d = ImageDraw.Draw(img)
    def xy(x,y): return ((x+15)/30*900, (15-y)/30*900)
    if path and len(path) > 1:
        d.line([xy(x,y) for x,y in path], fill=(235,40,40), width=3)
    if racks:
        order = visit_order or {}
        for r in racks:
            x,y = r['center']
            sx,sy=xy(x,y)
            # Rack IDs are always visible; when routing, prefix each selected rack
            # with its visit sequence so the displayed plan is easy to follow.
            label = f"{order[r['id']]}:{r['id']}" if r['id'] in order else r['id']
            d.text((sx+4,sy-10), label, fill=(0,90,210), stroke_width=1, stroke_fill='white')
    sx,sy=xy(0,0)
    d.ellipse((sx-7,sy-7,sx+7,sy+7), fill=(255,165,0), outline=(0,0,0), width=2)
    d.text((sx+8,sy+4), 'DOCK (0,0)', fill=(0,0,0), stroke_width=1, stroke_fill='white')
    return img



def seed_sample_maps():
    """Create reproducible axis-aligned sample warehouses for testing."""
    import numpy as _np
    samples = {}
    def base():
        g = _np.zeros((GRID_N, GRID_N), dtype=bool)
        g[8:12, :] = True; g[-12:-8, :] = True
        g[:, 8:12] = True; g[:, -12:-8] = True
        # Leave a dock opening at the middle of the south wall.
        g[8:12, 142:158] = False
        return g
    # Rectangular racks with broad lanes; best for verifying straight sweeps.
    g=base()
    for x in (48, 92, 136, 180, 224):
        g[48:78, x:x+24] = True
        g[104:134, x:x+24] = True
        g[160:190, x:x+24] = True
    samples['sample_open_aisles'] = g
    # Long parallel rack rows, with clear cross-aisles.
    g=base()
    for y in (42, 82, 122, 162, 202, 242):
        g[y:y+12, 38:112] = True
        g[y:y+12, 142:216] = True
        g[y:y+12, 246:278] = True
    samples['sample_parallel_rows'] = g
    # Narrower aisles and alternating rack lengths.
    g=base()
    for y in (38, 82, 126, 170, 214):
        g[y:y+20, 42:110] = True
        g[y:y+20, 126:194] = True
        g[y:y+20, 210:278] = True
    # Cross aisle gaps by removing selected small bands from rack ends.
    g[82:102, 42:54] = False; g[170:190, 266:278] = False
    samples['sample_dense_aisles'] = g
    # A few separated obstacles, not a regular warehouse: test recovery.
    g=base()
    for x,y,w,h in [(42,45,28,42),(96,55,50,18),(174,40,26,55),(224,68,42,26),
                    (55,130,18,54),(102,118,44,24),(170,135,54,18),(235,150,22,50),
                    (65,218,52,18),(142,205,24,42),(190,235,60,18)]:
        g[y:y+h,x:x+w]=True
    samples['sample_irregular_obstacles'] = g
    for name, grid in samples.items():
        path = MAP_DIR / f'{name}.json'
        if not path.exists(): save_map(name, grid)
        png = MAP_DIR / f'{name}.png'
        if not png.exists(): grid_png(grid).save(png)

seed_sample_maps()

@app.route('/')
def index(): return render_template('index.html')

@app.get('/api/maps')
def maps_list():
    result=[]
    for p in sorted(MAP_DIR.glob('*.json')):
        try:
            data=json.loads(p.read_text()); result.append({'name':data.get('name',p.stem)})
        except Exception: pass
    return jsonify(result)

@app.post('/api/maps')
def maps_save():
    body=request.get_json(force=True)
    name=safe_name(body.get('name'))
    rows=body.get('rows')
    if not isinstance(rows,list) or len(rows)!=GRID_N or any(not isinstance(r,str) or len(r)!=GRID_N or set(r)-{'0','1'} for r in rows):
        return jsonify(error='Expected a 300 x 300 grid of 0/1 cells.'),400
    grid=np.array([[c=='1' for c in row] for row in rows],dtype=bool)
    save_map(name,grid)
    img=grid_png(grid)
    img.save(MAP_DIR/f'{name}.png')
    return jsonify(ok=True,name=name)

@app.get('/api/maps/<name>')
def map_get(name):
    grid = load_map(name)
    if grid is None: return jsonify(error='Map not found'), 404
    return jsonify(name=safe_name(name), rows=[''.join('1' if c else '0' for c in row) for row in grid])

@app.get('/api/maps/<name>.png')
def map_png(name):
    p=MAP_DIR/f'{safe_name(name)}.png'
    if not p.exists():
        grid=load_map(name)
        if grid is None: return 'Map not found',404
        grid_png(grid).save(p)
    return send_file(p, mimetype='image/png', as_attachment=True, download_name=f'{safe_name(name)}.png')

@app.post('/api/racks')
def racks_api():
    body=request.get_json(force=True); grid=load_map(body.get('name',''))
    if grid is None: return jsonify(error='Map not found'),404
    racks, walls=identify_racks(grid)
    img=grid_png(grid,racks=racks)
    out=OUT_DIR/f'{safe_name(body.get("name"))}_rack_ids.png'; img.save(out)
    return jsonify(racks=racks, walls=walls, image=f'/api/preview/{out.name}', download=f'/api/download/{out.name}')

@app.get('/api/preview/<filename>')
def preview(filename):
    p = OUT_DIR / Path(filename).name
    if not p.exists(): return 'File not found', 404
    return send_file(p, mimetype='image/png', as_attachment=False)

@app.get('/api/download/<filename>')
def download(filename):
    name=Path(filename).name
    p=OUT_DIR/name
    if not p.exists(): return 'File not found',404
    return send_file(p, as_attachment=True, download_name=name)

@app.post('/api/slam/start')
def slam_start():
    body=request.get_json(force=True); name=safe_name(body.get('name',''))
    truth=load_map(name)
    if truth is None: return jsonify(error='Choose a saved map first.'),404
    run_id=f'{name}_{int(time.time()*1000)}'
    sim=SLAMSimulation(truth)
    with STATE_LOCK: RUNS[run_id]={'sim':sim,'done':False,'error':None,'frames':0,'name':name}
    Thread(target=run_sim, args=(run_id,), daemon=True).start()
    return jsonify(run_id=run_id)

def run_sim(run_id):
    with STATE_LOCK: entry=RUNS[run_id]; sim=entry['sim']
    try:
        max_steps=3600
        for i in range(max_steps):
            sim.step()
            # Pace the simulation so live rover/path/LiDAR updates are visible.
            time.sleep(0.025)
            if i % 4 == 0:
                with STATE_LOCK: entry['frames']=i+1
            # Stop when the reachable free space is well explored or the step budget is exhausted.
            if i > 500 and (sim.no_frontiers or sim.coverage_estimate() >= 0.90): break
        sim.return_to_dock()
        out=OUT_DIR/f'{run_id}_slam.png'
        grid_png(sim.known_occupied(), path=sim.path, pose=sim.pose).save(out)
        with STATE_LOCK:
            entry['done']=True; entry['frames']=sim.steps; entry['result']=str(out); entry['summary']=sim.summary()
    except Exception as e:
        with STATE_LOCK: entry['done']=True; entry['error']=str(e)

@app.get('/api/slam/status/<run_id>')
def slam_status(run_id):
    with STATE_LOCK:
        e=RUNS.get(run_id)
        if not e: return jsonify(error='Run not found'),404
        sim=e['sim']; done=e['done']; error=e['error']; frames=e['frames']; summary=e.get('summary')
        # Live image includes the current rover, red trajectory and green LiDAR rays.
        buf=io.BytesIO(); slam_live_png(sim).save(buf,format='PNG')
        png=base64.b64encode(buf.getvalue()).decode('ascii')
        pose=[round(float(v),2) for v in sim.pose]
        cov=round(100*sim.coverage_estimate(),1)
        result={'done':done,'error':error,'step':sim.steps,'mapped_cells':int(sim.known_mask().sum()),'coverage':cov,'pose':pose,'mode':sim.mode,'frontiers':sim.frontier_count,'centering_mode':sim.centering_mode,'center_error_m':sim.center_error_m,'left_wall_m':sim.left_wall_m,'right_wall_m':sim.right_wall_m,'image':'data:image/png;base64,'+png,'summary':summary}
        if done and e.get('result'): result['download']='/api/download/'+Path(e['result']).name
        return jsonify(result)

@app.post('/api/route/start')
def route_start():
    body=request.get_json(force=True); name=safe_name(body.get('name','')); grid=load_map(name)
    if grid is None: return jsonify(error='Choose a saved map.'),404
    racks,_=identify_racks(grid)
    wanted=list(dict.fromkeys(s.strip().upper() for s in (body.get('ids','').split(',')) if s.strip()))
    by_id={r['id']:r for r in racks}
    missing=[x for x in wanted if x not in by_id]
    if missing: return jsonify(error='Unknown rack ID(s): '+', '.join(missing), available=list(by_id)),400
    # Empty input means visit every detected rack; explicit IDs retain user selection.
    if not wanted:
        wanted = [r['id'] for r in racks]
    if not wanted:
        return jsonify(error='No racks were detected on this map. Run Identify racks and check the map first.'),400
    route=optimize_route([by_id[x] for x in wanted])
    try:
        full_path, visits = plan_rack_visits(grid, route)
    except ValueError as exc:
        return jsonify(error=str(exc)),400
    out=OUT_DIR/f'{name}_rack_visit_route.png'
    visit_order={rack['id']: index for index, rack in enumerate(route, 1)}
    grid_png(grid, racks=racks, path=[cell_to_world_xy(p) for p in full_path], visit_order=visit_order).save(out)
    # This is a grid-planned simulated route preview. It does not drive physical hardware.
    return jsonify(ok=True, route=[r['id'] for r in route], visits=visits,
                   path_cells=len(full_path), distance_m=round(max(0,len(full_path)-1)*0.1,2),
                   image=f'/api/preview/{out.name}', download=f'/api/download/{out.name}',
                   message='Collision-aware grid path planned from the dock to a reachable approach point at each rack, then back to the dock. Red line shows the planned route; selected rack labels are prefixed with visit order (1:R01, 2:R03, etc.). This is a simulation/preview, not physical robot control.')

def cell_to_world_xy(cell):
    ix, iy = cell
    return (ix * 0.1 - 15.0 + 0.05, iy * 0.1 - 15.0 + 0.05)

def astar_free(grid, start, goal):
    """4-connected A* through free cells; grid coordinates are (x, y)."""
    h, w = grid.shape
    def free(p):
        x,y=p
        return 0 <= x < w and 0 <= y < h and not grid[y,x]
    if not free(start) or not free(goal): return None
    frontier=[(abs(start[0]-goal[0])+abs(start[1]-goal[1]),0,start)]
    came={}; cost={start:0}; closed=set()
    while frontier:
        _,g,cur=heapq.heappop(frontier)
        if cur in closed: continue
        if cur==goal:
            path=[cur]
            while cur in came: cur=came[cur]; path.append(cur)
            return list(reversed(path))
        closed.add(cur)
        x,y=cur
        for nxt in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            if not free(nxt) or nxt in closed: continue
            ng=g+1
            if ng < cost.get(nxt,1e12):
                cost[nxt]=ng; came[nxt]=cur
                heuristic=abs(nxt[0]-goal[0])+abs(nxt[1]-goal[1])
                heapq.heappush(frontier,(ng+heuristic,ng,nxt))
    return None

def rack_approach_cells(grid, rack):
    """Return reachable candidates around a rack bounding box, expanding outward.

    Include multiple perimeter rings rather than stopping at the first ring:
    the nearest free cells can be trapped behind another rack or a wall.
    A* later filters candidates that are not actually reachable.
    """
    bx, by, bw, bh = rack.get('bbox_m', [rack['center'][0], rack['center'][1], 0.5, 0.5])
    h, w = grid.shape
    x0 = max(0, min(w-1, int((bx + 15.0) / 0.1)))
    y0 = max(0, min(h-1, int((by + 15.0) / 0.1)))
    x1 = max(x0, min(w-1, int((bx + bw + 15.0) / 0.1)))
    y1 = max(y0, min(h-1, int((by + bh + 15.0) / 0.1)))
    candidates = []
    for radius in range(1, 16):
        xa, xb = max(0, x0-radius), min(w-1, x1+radius)
        ya, yb = max(0, y0-radius), min(h-1, y1+radius)
        ring = []
        for x in range(xa, xb+1):
            ring.extend(((x, ya), (x, yb)))
        for y in range(ya+1, yb):
            ring.extend(((xa, y), (xb, y)))
        for x, y in ring:
            if not grid[y, x]:
                candidates.append((x, y))
    return list(dict.fromkeys(candidates))

def astar_to_any(grid, start, goals):
    """Find the shortest 4-connected route from start to any safe approach cell."""
    h, w = grid.shape
    goals = set(goals)
    if not goals:
        return None, None
    sx, sy = start
    if not (0 <= sx < w and 0 <= sy < h) or grid[sy, sx]:
        return None, None
    frontier=[start]; head=0
    came={}; seen={start}
    while head < len(frontier):
        cur=frontier[head]; head += 1
        if cur in goals:
            path=[cur]
            while cur in came:
                cur=came[cur]; path.append(cur)
            path.reverse()
            return path, path[-1]
        x,y=cur
        for nxt in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            nx,ny=nxt
            if nxt in seen or nx<0 or nx>=w or ny<0 or ny>=h or grid[ny,nx]:
                continue
            seen.add(nxt); came[nxt]=cur; frontier.append(nxt)
    return None, None

def plan_rack_visits(grid, route):
    start=(int(15/0.1),int(15/0.1))
    current=start; complete=[]; visits=[]
    for rack in route:
        candidates=rack_approach_cells(grid,rack)
        best_path,best_goal=astar_to_any(grid,current,candidates)
        if best_path is None:
            raise ValueError(f"No collision-free path to rack {rack['id']}. Check that the rack has a free aisle beside it and the dock is connected.")
        complete.extend(best_path if not complete else best_path[1:])
        visits.append({'id':rack['id'],'approach_m':[round(best_goal[0]*0.1-15+0.05,2),round(best_goal[1]*0.1-15+0.05,2)]})
        current=best_goal
    back=astar_free(grid,current,start)
    if back is None: raise ValueError('Could not find a safe path back to the dock.')
    complete.extend(back[1:] if complete else back)
    return complete,visits

def optimize_route(racks):
    if not racks: return []
    def dist(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])
    dock=(0.0,0.0)
    if len(racks)<=8:
        best=None; bestd=float('inf')
        for perm in __import__('itertools').permutations(racks):
            d=dist(dock,perm[0]['center'])+sum(dist(perm[i]['center'],perm[i+1]['center']) for i in range(len(perm)-1))+dist(perm[-1]['center'],dock)
            if d<bestd: bestd=d; best=perm
        return list(best)
    remaining=racks[:]; out=[]; p=dock
    while remaining:
        r=min(remaining,key=lambda q:dist(p,q['center'])); out.append(r); remaining.remove(r); p=r['center']
    return out

if __name__=='__main__':
    app.run(host='127.0.0.1',port=5000,debug=False,threaded=True)
