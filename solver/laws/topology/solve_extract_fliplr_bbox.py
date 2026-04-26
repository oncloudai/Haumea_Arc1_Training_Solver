import numpy as np
from typing import List, Optional

def solve_grid_7468f01a(grid):
    grid = np.array(grid)
    coords = np.argwhere(grid > 0)
    if coords.size == 0: return grid
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    subgrid = grid[r_min : r_max + 1, c_min : c_max + 1]
    return np.fliplr(subgrid)

def solve_extract_fliplr_bbox(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_7468f01a(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_7468f01a(ti) for ti in solver.test_in]
    return None
