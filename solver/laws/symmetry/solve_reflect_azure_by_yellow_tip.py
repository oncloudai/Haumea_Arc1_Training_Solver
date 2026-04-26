import numpy as np
from typing import List, Optional

def solve_grid_760b3cac(grid):
    grid = np.array(grid)
    h, w = grid.shape
    yellow_coords = np.argwhere(grid == 4)
    if yellow_coords.size == 0: return grid
    r_h = int(np.median(yellow_coords[:, 0]))
    c_h = int(np.median(yellow_coords[:, 1]))
    
    tip_coords = yellow_coords[yellow_coords[:, 0] == r_h - 1]
    if tip_coords.size > 0:
        c_tip = tip_coords[0, 1]
        if c_tip < c_h: axis = c_h - 1.5
        elif c_tip > c_h: axis = c_h + 1.5
        else: axis = c_h
    else:
        axis = c_h
        
    out = grid.copy()
    azure_coords = np.argwhere(grid == 8)
    for r, c in azure_coords:
        c_ref = int(2 * axis - c)
        if 0 <= c_ref < w:
            out[r, c_ref] = 8
    return out

def solve_reflect_azure_by_yellow_tip(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_760b3cac(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_760b3cac(ti) for ti in solver.test_in]
    return None
