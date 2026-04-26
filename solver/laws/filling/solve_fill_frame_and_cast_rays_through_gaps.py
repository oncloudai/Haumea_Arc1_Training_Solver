import numpy as np
from typing import List, Optional

def solve_grid_d4f3cd78(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    c5_pixels = np.argwhere(grid == 5)
    if len(c5_pixels) == 0: return grid
    r_min, r_max = c5_pixels[:, 0].min(), c5_pixels[:, 0].max()
    c_min, c_max = c5_pixels[:, 1].min(), c5_pixels[:, 1].max()
    output_grid = grid.copy()
    for r in range(r_min + 1, r_max):
        for c in range(c_min + 1, c_max):
            if grid[r, c] == 0: output_grid[r, c] = 8
    for c in range(c_min + 1, c_max):
        if grid[r_min, c] == 0:
            output_grid[r_min, c] = 8
            for r in range(r_min - 1, -1, -1): output_grid[r, c] = 8
        if grid[r_max, c] == 0:
            output_grid[r_max, c] = 8
            for r in range(r_max + 1, h): output_grid[r, c] = 8
    for r in range(r_min + 1, r_max):
        if grid[r, c_min] == 0:
            output_grid[r, c_min] = 8
            for c_ in range(c_min - 1, -1, -1): output_grid[r, c_] = 8
        if grid[r, c_max] == 0:
            output_grid[r, c_max] = 8
            for c_ in range(c_max + 1, w): output_grid[r, c_] = 8
    return output_grid

def solve_fill_frame_and_cast_rays_through_gaps(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_d4f3cd78(inp)
        if not np.array_equal(res, out):
            consistent = False; break
    if consistent:
        return [solve_grid_d4f3cd78(ti) for ti in solver.test_in]
    return None
