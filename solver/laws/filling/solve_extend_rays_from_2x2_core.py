import numpy as np
from typing import List, Optional

def solve_grid_7ddcd7ec(grid):
    grid = np.array(grid)
    h, w = grid.shape
    out = grid.copy()
    unique_colors = np.unique(grid)
    unique_colors = unique_colors[unique_colors > 0]
    if unique_colors.size == 0: return grid
    color = unique_colors[0]
    coords = np.argwhere(grid == color)
    r_core, c_core = -1, -1
    for r, c in coords:
        if (r+1 < h and c+1 < w and grid[r+1, c] == color and grid[r, c+1] == color and grid[r+1, c+1] == color):
            r_core, c_core = r, c; break
    if r_core == -1: return grid
    core_coords = {(r_core, c_core), (r_core+1, c_core), (r_core, c_core+1), (r_core+1, c_core+1)}
    for r, c in coords:
        if (r, c) in core_coords: continue
        dr, dc = 0, 0
        if r < r_core: dr = -1
        elif r > r_core + 1: dr = 1
        if c < c_core: dc = -1
        elif c > c_core + 1: dc = 1
        if dr != 0 or dc != 0:
            curr_r, curr_c = r, c
            while 0 <= curr_r < h and 0 <= curr_c < w:
                out[curr_r, curr_c] = color
                curr_r += dr; curr_c += dc
    return out

def solve_extend_rays_from_2x2_core(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_7ddcd7ec(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_7ddcd7ec(ti) for ti in solver.test_in]
    return None
