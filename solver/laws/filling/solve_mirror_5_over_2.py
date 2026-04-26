import numpy as np
from typing import List, Optional

def solve_grid_f8a8fe49(input_grid):
    grid = np.array(input_grid)
    rows, cols = grid.shape
    
    # 1. Find all color 2 and color 5 pixels
    c2_pixels = np.argwhere(grid == 2)
    c5_pixels = np.argwhere(grid == 5)
    
    output_grid = grid.copy()
    # Remove original 5s
    for r, c in c5_pixels:
        output_grid[r, c] = 0
        
    for r, c in c5_pixels:
        # Find nearest c2 pixel in same row or column
        min_dist = float('inf')
        target = None
        
        for nr, nc in c2_pixels:
            if nr == r:
                dist = abs(nc - c)
                if dist < min_dist:
                    min_dist = dist
                    target = (r, nc - (c - nc))
            elif nc == c:
                dist = abs(nr - r)
                if dist < min_dist:
                    min_dist = dist
                    target = (nr - (r - nr), c)
        
        if target:
            tr, tc = target
            if 0 <= tr < rows and 0 <= tc < cols:
                output_grid[tr, tc] = 5
                
    return output_grid

def solve_mirror_5_over_2(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_f8a8fe49(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_f8a8fe49(ti) for ti in solver.test_in]
    return None
