import numpy as np
from typing import List, Optional

def solve_grid_dae9d2b5(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    if w != 6: return grid
    left = grid[:, 0:3]
    right = grid[:, 3:6]
    output_grid = np.zeros((3, 3), dtype=int)
    for r in range(3):
        for c in range(3):
            if left[r, c] != 0 or right[r, c] != 0: output_grid[r, c] = 6
    return output_grid

def solve_3x3_logical_or_to_color_6(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        if inp.shape != (3, 6) or out.shape != (3, 3): consistent = False; break
        res = solve_grid_dae9d2b5(inp)
        if not np.array_equal(res, out):
            consistent = False; break
    if consistent:
        return [solve_grid_dae9d2b5(ti) for ti in solver.test_in]
    return None
