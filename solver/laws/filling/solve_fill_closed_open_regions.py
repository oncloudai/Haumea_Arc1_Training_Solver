import numpy as np
from scipy.ndimage import label
from typing import List, Optional

def solve_grid_7b6016b9(grid):
    grid = np.array(grid)
    h, w = grid.shape
    unique_colors = np.unique(grid)
    unique_colors = unique_colors[unique_colors > 0]
    if unique_colors.size == 0: return grid
    mask = (grid == 0)
    labeled, num_features = label(mask)
    out = grid.copy()
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled == i)
        touches_edge = False
        for r, c in coords:
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                touches_edge = True; break
        fill_color = 3 if touches_edge else 2
        for r, c in coords: out[r, c] = fill_color
    return out

def solve_fill_closed_open_regions(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_7b6016b9(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_7b6016b9(ti) for ti in solver.test_in]
    return None
