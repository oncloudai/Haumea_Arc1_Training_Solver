import numpy as np
from typing import List, Optional

def solve_grid_a65b410d(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    
    # Find color 2
    r_idx = -1
    c_start = -1
    n = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 2:
                if r_idx == -1:
                    r_idx = r
                    c_start = c
                n += 1
    
    if r_idx == -1:
        return grid
    
    output_grid = grid.copy()
    
    # Color 3: rows before R
    for i in range(r_idx):
        width = r_idx + n - i
        for c in range(min(width, w)):
            output_grid[i, c] = 3
            
    # Color 1: rows after R
    for j in range(1, n):
        row = r_idx + j
        if row < h:
            width = n - j
            for c in range(min(width, w)):
                output_grid[row, c] = 1
                
    return output_grid

def solve_color_rows_relative_to_marker(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_a65b410d(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_a65b410d(ti) for ti in solver.test_in]
    return None
