# Warehouse Rover Lab — Python web app

A local Flask web application for creating 30 m × 30 m occupancy maps, testing a simulated 2-D LiDAR/occupancy-grid SLAM pipeline, assigning rack candidate IDs, and planning rack visit order.

## Run (Ubuntu / Linux / Windows)

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in your browser. Maps are stored in `data/maps/`; run images in `data/runs/`.

## Map editor

- Grid: 300 × 300 cells at 0.1 m per cell (30 m × 30 m).
- Left drag: occupied cells.
- Right drag: free cells.
- Middle drag: pan.
- Mouse wheel: zoom around pointer.
- Save maps by name; map PNGs are downloadable.
- Dock pose is (0,0), the centre of the map. It is not itself an occupied obstacle because the rover must be able to start there. The first LiDAR scan maps any obstacles around the dock.

## SLAM model

`slam.py` simulates 120 LiDAR rays, noisy odometry, log-odds occupancy updates, and conservative local scan-to-map correction. The explorer now follows map-derived frontier goals using A* over the partial occupancy map; mapped-free cells are cheaper than unknown cells, and a per-step simulated collision guard rejects moves when the map misses an obstacle. This prevents the old turn-only policy from automatically following the outside boundary at aisle ends. For centreline control, LiDAR endpoint geometry is checked to ensure a side return extends along the aisle rather than being a rack end face. When both side surfaces are visible, the controller uses the difference between measured distances to steer toward their midpoint. One-wall fallback is only enabled briefly after both sides have confirmed a real aisle; isolated walls in cross-aisles do not cause a centreline command. The SLAM panel reports the centring mode, estimated centreline offset, and current left/right distances. Scan matching searches a smaller neighbourhood and accepts a pose correction only when it meaningfully improves the scan score, reducing arbitrary drift along long, featureless aisles. Local collision checks use ground truth as a simulation safety guard, which would not be available to a physical robot. Rectangle filling remains heuristic: partial/noisy outlines may not be filled, and a wrongly inferred rectangle can hide free space. This is an educational prototype, not a validated production SLAM system; compare reconstructed maps against ground truth before relying on results.

## Rack IDs

Rack identification is currently heuristic connected-component segmentation of occupied cells. It classifies compact occupied components as rack candidates and elongated components as likely walls. **No AI model is trained.** Because the map editor has only occupied/free labels, touching racks/walls may merge and confuse classification. For real ML training, add labeled examples (wall/rack/other) and train/validate a classifier or segmentation model.

## Route planning

Rack IDs can be submitted as comma-separated values. Up to 8 rack candidates are ordered by exact permutation search to reduce approximate travel distance; larger lists use nearest-neighbour ordering. For each rack, the planner finds a free approach cell and uses 4-connected A* over the saved occupancy grid, avoiding occupied cells, then plans a return to the dock. The UI displays and allows downloading the red route overlay, along with approach coordinates and estimated grid-path length. Rack identification remains a geometric heuristic; touching racks/walls can be merged. This is a software route preview, not a live animated or physical robot controller. Validate map geometry, clearance, localization, and emergency stopping before any real-world deployment.

## Evaluation tips

Download both the ground-truth map and the reconstructed SLAM PNG. For quantitative evaluation, compare occupancy grids with intersection-over-union, precision/recall, and boundary distance. Unknown SLAM cells should be scored separately rather than automatically treated as free.

## OpenCV-based map interpretation

The SLAM simulator uses OpenCV (`opencv-python-headless`) to process the partial
occupancy map. Small morphological closing repairs gaps in noisy occupied outlines;
contours and contour hierarchy are used to identify thin wall-like components and
closed rectangular rack outlines. Closed rack interiors are marked as inferred
occupied cells and are treated as collision-forbidden by the simulated controller.
The direction chooser combines the current map's open cardinal runs, the amount of
unknown space along each run, and the latest LiDAR rays. When scores tie, it prefers
a perpendicular turn from the previous heading. This is a heuristic CV pipeline,
not a trained model, and inferred rack labels should be checked against the known map.

## Regression tests

With dependencies installed, run the centring and pose-stability regression tests from this folder:

```bash
python -m unittest discover -s tests -v
```

The tests exercise left/right steering sign, no correction in centred/open space, and convergence inside a synthetic aisle while checking that the rover does not collide.

### OpenCV fallback for missing SLAM frontiers

When the normal sampled frontier selector cannot find a target, the SLAM planner now uses OpenCV connected-component analysis on the *unknown portion of the estimated occupancy grid*. It ranks sizeable unknown regions and chooses a nearby known-free boundary cell as a fallback goal. The A* planner then routes to that free approach cell; the rover does not aim directly into an unknown region. This fallback uses the partial SLAM map, not the simulation's ground-truth map.

Regression tests (including the OpenCV fallback) can be run with:

```bash
python -m unittest discover -s tests -v
```


## Identify racks and visit racks (completed workflow)

1. Save or choose a map in the editor.
2. Open **Identify racks**, select that same map, and click **Identify racks**. The result lists stable IDs (`R01`, `R02`, …), approximate centre/size, and a downloadable annotated PNG.
3. Open **Visit racks** and select the same map. Enter comma-separated IDs to choose a subset (e.g. `R01, R03`). Leave the field blank to plan a visit to every detected rack.
4. The planner orders the selected racks to reduce estimated travel, uses 4-connected A* to find free approach cells around each rack, and adds a final A* leg back to the dock. If a nearby approach cell is blocked off, it also checks farther perimeter rings for an accessible alternative.
5. The result shows visit order, approach coordinates, total path length, and a downloadable PNG with the planned route in red.

This is a planned route preview, not a live animated visit simulation or hardware controller. The rack detector is shape-based OpenCV segmentation; if racks touch walls or each other, verify the IDs and map before using the route.
