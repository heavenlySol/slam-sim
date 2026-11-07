import ast
import heapq
import math
import unittest
from pathlib import Path
import numpy as np
from slam import identify_racks, N

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

def load_route_functions():
    tree = ast.parse(APP_PATH.read_text())
    names = {"cell_to_world_xy", "astar_free", "astar_to_any", "rack_approach_cells", "plan_rack_visits", "optimize_route"}
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    env = {"np": np, "math": math, "heapq": heapq}
    exec(compile(module, str(APP_PATH), "exec"), env)
    return env

class RackWorkflowTests(unittest.TestCase):
    def test_detect_and_plan_return_to_dock(self):
        grid = np.zeros((N, N), dtype=bool)
        grid[8:12, :] = True
        grid[-12:-8, :] = True
        grid[:, 8:12] = True
        grid[:, -12:-8] = True
        grid[8:12, 142:158] = False
        grid[50:80, 50:72] = True
        grid[110:140, 100:125] = True
        grid[190:215, 210:245] = True
        racks, _ = identify_racks(grid)
        self.assertEqual(len(racks), 3)
        funcs = load_route_functions()
        route = funcs["optimize_route"](racks)
        path, visits = funcs["plan_rack_visits"](grid, route)
        self.assertEqual(len(visits), 3)
        self.assertEqual(path[0], (150, 150))
        self.assertEqual(path[-1], (150, 150))
        self.assertTrue(all(not grid[y, x] for x, y in path))

if __name__ == "__main__":
    unittest.main()
