import numpy as np
from typing import List, Optional

def solve_grid_a699fb00(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    output_grid = grid.copy()
    
    # Horizontal pairs ONLY
    for r in range(h):
        for c in range(w - 2):
            if grid[r, c] == 1 and grid[r, c+2] == 1:
                output_grid[r, c+1] = 2
                
    return output_grid

def solve_fill_horizontal_gap_between_1s(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_a699fb00(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_a699fb00(ti) for ti in solver.test_in]
    return None
