import numpy as np
from typing import List, Optional

def solve_grid_cf98881b(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    if h != 4 or w != 14:
        return grid
    
    # 4x14 input. Separators (color 2) at col 4 and 9.
    s1 = grid[:, 0:4]
    s2 = grid[:, 5:9]
    s3 = grid[:, 10:14]
    
    output_grid = np.zeros((4, 4), dtype=int)
    # Priority: 4 > 9 > 1
    priority = [4, 9, 1]
    
    for r in range(4):
        for c in range(4):
            vals = [s1[r, c], s2[r, c], s3[r, c]]
            non_zero_vals = [v for v in vals if v != 0]
            
            if not non_zero_vals:
                output_grid[r, c] = 0
            else:
                found = False
                for p in priority:
                    if p in non_zero_vals:
                        output_grid[r, c] = p
                        found = True
                        break
                if not found:
                    output_grid[r, c] = non_zero_vals[0]
                    
    return output_grid

def solve_three_sections_priority_overlay_4x4(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        if inp.shape != (4, 14) or out.shape != (4, 4):
            consistent = False; break
        res = solve_grid_cf98881b(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_cf98881b(ti) for ti in solver.test_in]
    return None
