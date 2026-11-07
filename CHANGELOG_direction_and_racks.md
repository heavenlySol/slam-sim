# Navigation / rack inference changes

- When a movement target is blocked, score the four cardinal directions using open clearance and unknown cells in the SLAM map.
- Prefer a straight continuation when it is the best safe option; on a close tie, prefer a perpendicular direction to the previous heading, choosing randomly between equally good perpendicular options.
- Keep movement axis-aligned rather than diagonally cutting through aisles.
- Every 12 simulation steps, inspect occupied SLAM components for small axis-aligned, mostly closed rectangular outlines. Fill a qualifying outline's interior as inferred rack cells.
- Render inferred rack cells as occupied and treat them as forbidden during simulated collision checks.
- The detector is deliberately heuristic and must be checked against ground truth; this is not a production-certified SLAM or semantic rack detector.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

## Obstacle-triggered straight-line controller (2026-10-12)

- The rover now maintains a cardinal direction and advances in 10 cm steps while
  its next step is clear. Frontier scoring cannot trigger a turn during a clear run.
- A blocked forward step triggers a direction decision. Safe directions are ranked
  by unexplored cells ahead and clear run length; a perpendicular option breaks ties.
  Reverse is avoided when another safe direction exists, and remaining ties are seeded-random.
- Turns happen in place before forward motion resumes; movement remains axis-aligned.
- OpenCV rack inference and occupancy-map updates remain enabled.

## Dual left/right wall-centering controllers (2026-10-12)

- Added separate left-wall and right-wall LiDAR controllers.
- For a nominal 2.0 m aisle and 0.50 m rover, the target is 1.0 m from the rover/LiDAR centre to each wall (about 0.75 m body clearance on each side).
- When both walls are visible, controller corrections are fused; when only one is visible, the corresponding side controller is used.
- Corrections are capped and have a deadband to reduce oscillation. They do not change the cardinal heading or override front-obstacle turns.
- Fixed application of lateral correction so the corrected position is written to the simulated actual pose before odometry/map updates.

## SLAM navigation centering and pose-stability update

- Replaced independently clipped left/right wall corrections with direct midpoint estimation from both reliable side-LiDAR distances, allowing the controller to handle aisles that are not exactly 2.0 m wide.
- Kept a cautious nominal-width fallback for a single nearby wall and disabled correction for weak returns/open spaces.
- Changed centering movement from instantaneous sideways position shifts to bounded, smooth heading changes during forward motion.
- Reduced per-step odometry noise and constrained scan matching to small corrections that must improve the scan score, preventing large unobserved drift along long straight aisles.
- Added SLAM-panel diagnostics for the centering mode, measured offset, and left/right distance readings.
- Added deterministic regression tests for correction direction, open-space behavior, and convergence in a synthetic corridor.
- Rack identification and rack-visit planning behavior were not changed in this update.


## Top-section / frontier navigation fix (2026-10-13)

- Connected the previously unused frontier target to actual movement using A* over the partial SLAM occupancy map. The planner prefers known-free cells and assigns higher cost to unknown cells.
- Replanned on a frontier arrival, a stale route, or a blocked step. A collision detected by the simulator invalidates the route so a new scan can inform the next plan.
- Turn selection now follows the direction that makes progress toward the current frontier when safe, including reversing down an aisle when that is better than turning along the outer perimeter.
- LiDAR side distances now use local hit-endpoint line geometry. Transverse rack end faces and corner clusters are rejected rather than treated as aisle walls.
- Single-wall centring is only allowed briefly after two-sided observations have confirmed a corridor, reducing false steering near cross-aisles.
- Added regression coverage for rack-end rejection and for avoiding the previous outer-perimeter exploration loop. Rack IDs and rack-visit features were not modified.
