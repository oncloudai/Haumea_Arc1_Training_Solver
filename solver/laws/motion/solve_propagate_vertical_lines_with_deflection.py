import numpy as np
from typing import List, Optional

def solve_grid_d9f24cd1(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    c2_cols = np.where(grid[h-1, :] == 2)[0]
    output_grid = grid.copy()
    for c_start in c2_cols:
        curr_c = c_start
        for r in range(h-1, -1, -1):
            if grid[r, curr_c] == 5:
                curr_c += 1
                if r + 1 < h: output_grid[r + 1, curr_c] = 2
            if output_grid[r, curr_c] == 0: output_grid[r, curr_c] = 2
    return output_grid

def solve_propagate_vertical_lines_with_deflection(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_d9f24cd1(inp)
        if not np.array_equal(res, out):
            consistent = False; break
    if consistent:
        return [solve_grid_d9f24cd1(ti) for ti in solver.test_in]
    return None
