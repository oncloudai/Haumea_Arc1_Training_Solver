import numpy as np
from typing import List, Optional

def solve_grid_75b8110e(grid):
    grid = np.array(grid)
    if grid.shape != (8, 8): return grid
    q1 = grid[0:4, 0:4]
    q2 = grid[0:4, 4:8]
    q3 = grid[4:8, 0:4]
    q4 = grid[4:8, 4:8]
    priority = [5, 6, 9, 4, 0]
    out = np.zeros((4, 4), dtype=int)
    for r in range(4):
        for c in range(4):
            colors = [q1[r, c], q2[r, c], q3[r, c], q4[r, c]]
            for p in priority:
                if p in colors:
                    out[r, c] = p
                    break
    return out

def solve_quadrant_priority_overlap_4x4(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        if inp.shape != (8, 8) or out.shape != (4, 4):
            consistent = False; break
        res = solve_grid_75b8110e(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_75b8110e(ti) for ti in solver.test_in]
    return None
