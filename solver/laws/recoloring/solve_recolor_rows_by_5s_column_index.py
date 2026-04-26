import numpy as np
from typing import List, Optional

def solve_grid_a85d4709(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    output_grid = np.zeros_like(grid)
    
    mapping = {0: 2, 1: 4, 2: 3}
    
    for r in range(h):
        # Find column of color 5
        cols = np.where(grid[r, :] == 5)[0]
        if len(cols) > 0:
            c = cols[0]
            val = mapping.get(c, 0)
            output_grid[r, :] = val
            
    return output_grid

def solve_recolor_rows_by_5s_column_index(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_a85d4709(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_a85d4709(ti) for ti in solver.test_in]
    return None
