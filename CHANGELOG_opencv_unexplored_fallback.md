# OpenCV unexplored-region fallback

- Added `_opencv_guess_unexplored_target()` in `slam.py`.
- When normal frontier sampling yields no candidates, OpenCV closes tiny holes, groups unknown cells, and ranks reachable known-free cells adjacent to unknown regions.
- The target is a free boundary/approach cell, not an unknown cell; existing A* planning and per-step collision checks remain in control.
- Added regression tests for fallback target selection and free-cell safety.
- Rack identification and Visit Racks behavior were not changed.
