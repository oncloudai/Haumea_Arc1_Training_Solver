import numpy as np
from typing import List, Optional

def solve_grid_f5b8619d(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    modified_grid = grid.copy()
    for c in range(w):
        col = grid[:, c]
        if np.any(col != 0):
            for r in range(h):
                if modified_grid[r, c] == 0:
                    modified_grid[r, c] = 8
                    
    output_grid = np.block([
        [modified_grid, modified_grid],
        [modified_grid, modified_grid]
    ])
    return output_grid

def solve_tile_2x2_with_filled_cols(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_f5b8619d(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_f5b8619d(ti) for ti in solver.test_in]
    return None
