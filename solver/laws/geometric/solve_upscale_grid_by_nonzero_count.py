import numpy as np
from typing import List, Optional

def solve_grid_ac0a08a4(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    f = np.sum(grid != 0)
    if f == 0: return grid
    out_h, out_w = h * f, w * f
    output_grid = np.zeros((out_h, out_w), dtype=int)
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0:
                color = grid[r, c]
                output_grid[r*f:(r+1)*f, c*f:(c+1)*f] = color
    return output_grid

def solve_upscale_grid_by_nonzero_count(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_ac0a08a4(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_ac0a08a4(ti) for ti in solver.test_in]
    return None
