import numpy as np
from typing import List, Optional

def solve_grid_7447852a(grid):
    grid = np.array(grid)
    h, w = grid.shape
    out = grid.copy()
    
    # Shape A: anchored at (1, x)
    # Offsets relative to (1, x): (0,0), (1,0), (1,1), (1,-1)
    shape_a = [(0, 0), (1, 0), (1, 1), (1, -1)]
    # Shape B: anchored at (0, x)
    # Offsets relative to (0, x): (0,0), (0,1), (0,2), (1,1)
    shape_b = [(0, 0), (0, 1), (0, 2), (1, 1)]
    
    x = 0
    i = 0
    while x < w:
        if i % 2 == 0:
            # Place Shape A at (1, x)
            r_base, c_base = 1, x
            for dr, dc in shape_a:
                r, c = r_base + dr, c_base + dc
                if 0 <= r < h and 0 <= c < w:
                    out[r, c] = 4
            x += 5
        else:
            # Place Shape B at (0, x)
            r_base, c_base = 0, x
            for dr, dc in shape_b:
                r, c = r_base + dr, c_base + dc
                if 0 <= r < h and 0 <= c < w:
                    out[r, c] = 4
            x += 7
        i += 1
        
    return out

def solve_alternating_shape_stamping_h(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_7447852a(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_7447852a(ti) for ti in solver.test_in]
    return None
