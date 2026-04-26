import numpy as np
from typing import List, Optional

def solve_grid_73251a56(grid):
    grid = np.array(grid)
    h, w = grid.shape
    
    # We look for a vanishing point (r0, c0) such that the color 
    # of a pixel (r, c) depends only on the slope (c - c0) / (r - r0).
    
    best_r0, best_c0 = None, None
    max_consistency = -1
    
    # ARC grids are small, let's search a reasonable range.
    # The vanishing point is often just outside the grid.
    search_range = range(-15, 20)
    
    # Optimize: only check non-zero pixels
    coords = np.argwhere(grid > 0)
    if coords.size == 0: return grid
    
    for r0 in search_range:
        for c0 in search_range:
            dr = coords[:, 0] - r0
            dc = coords[:, 1] - c0
            slopes = np.arctan2(dc, dr)
            
            # Check consistency: pixels with similar slopes should have the same color.
            sort_idx = np.argsort(slopes)
            sorted_slopes = slopes[sort_idx]
            sorted_colors = grid[coords[sort_idx, 0], coords[sort_idx, 1]]
            
            # A simple measure: number of times adjacent pixels have the same color
            matches = np.sum(sorted_colors[:-1] == sorted_colors[1:])
            
            if matches > max_consistency:
                max_consistency = matches
                best_r0, best_c0 = r0, c0

    if best_r0 is None:
        return grid

    # Fill using the best vanishing point
    dr = coords[:, 0] - best_r0
    dc = coords[:, 1] - best_c0
    slopes = np.arctan2(dc, dr)
    colors = grid[coords[:, 0], coords[:, 1]]
    
    out = grid.copy()
    for r in range(h):
        for c in range(w):
            if out[r, c] == 0:
                s = np.arctan2(c - best_c0, r - best_r0)
                # Find the color of the pixel with the nearest slope
                idx = np.argmin(np.abs(slopes - s))
                out[r, c] = colors[idx]
                
    return out

def solve_vanishing_point_slope_fill(solver) -> Optional[List[np.ndarray]]:
    # Task 73251a56
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_73251a56(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_73251a56(ti) for ti in solver.test_in]
    return None
