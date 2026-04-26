import numpy as np
from typing import List, Optional

def solve_grid_af902bf9(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    output_grid = grid.copy()
    
    # Find all color 4 pixels
    c4_pixels = np.argwhere(grid == 4)
    c4_set = set(tuple(p) for p in c4_pixels)
    
    # Find all rectangles formed by color 4 pixels
    for i in range(len(c4_pixels)):
        for j in range(len(c4_pixels)):
            if i == j: continue
            
            r1, c1 = c4_pixels[i]
            r2, c2 = c4_pixels[j]
            
            if r1 < r2 and c1 < c2:
                # Check if other two corners exist
                if (r1, c2) in c4_set and (r2, c1) in c4_set:
                    # Fill interior
                    for r in range(r1 + 1, r2):
                        for c in range(c1 + 1, c2):
                            output_grid[r, c] = 2
                            
    return output_grid

def solve_fill_rectangles_by_corners(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_af902bf9(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_af902bf9(ti) for ti in solver.test_in]
    return None
