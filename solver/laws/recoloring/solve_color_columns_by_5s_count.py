import numpy as np
from typing import List, Optional

def solve_grid_a61f2674(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    counts = {}
    for c in range(w):
        count = np.sum(grid[:, c] == 5)
        if count > 0:
            counts[c] = count
    
    if not counts:
        return np.zeros_like(grid)
    
    sorted_cols = sorted(counts.keys(), key=lambda x: counts[x])
    min_col = sorted_cols[0]
    max_col = sorted_cols[-1]
    
    output_grid = np.zeros_like(grid)
    
    # Fill max col with 1
    max_height = counts[max_col]
    for r in range(h - max_height, h):
        output_grid[r, max_col] = 1
        
    # Fill min col with 2
    min_height = counts[min_col]
    for r in range(h - min_height, h):
        output_grid[r, min_col] = 2
        
    return output_grid

def solve_color_columns_by_5s_count(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_a61f2674(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_a61f2674(ti) for ti in solver.test_in]
    return None
