import numpy as np
from scipy.ndimage import label
from typing import List, Optional

def solve_grid_50846271(grid, L=2):
    grid = np.array(grid)
    H, W = grid.shape
    out = grid.copy()
    
    # 1. Identify red components
    red_mask = (grid == 2)
    labeled, num_features = label(red_mask, structure=np.ones((3,3)))
    
    # 2. For each component, find the BEST cross
    for i in range(1, num_features + 1):
        comp_coords = np.argwhere(labeled == i)
        r_min, c_min = comp_coords.min(axis=0)
        r_max, c_max = comp_coords.max(axis=0)
        
        best_centers = []
        max_red_overlap = -1
        min_invalid_count = float('inf')
        
        for r in range(max(0, r_min - L), min(H, r_max + L + 1)):
            for c in range(max(0, c_min - L), min(W, c_max + L + 1)):
                
                cross_pixels = []
                for k in range(-L, L + 1):
                    cross_pixels.append((r + k, c))
                    cross_pixels.append((r, c + k))
                cross_pixels = list(set(cross_pixels))
                
                red_overlap = 0
                invalid_count = 0
                for cr, cc in cross_pixels:
                    if 0 <= cr < H and 0 <= cc < W:
                        val = grid[cr, cc]
                        if val == 2 and labeled[cr, cc] == i:
                            red_overlap += 1
                        elif val != 5 and val != 2:
                            invalid_count += 1
                    else:
                        invalid_count += 1
                
                if red_overlap > 0:
                    if red_overlap > max_red_overlap:
                        max_red_overlap = red_overlap
                        min_invalid_count = invalid_count
                        best_centers = [(r, c, cross_pixels)]
                    elif red_overlap == max_red_overlap:
                        if invalid_count < min_invalid_count:
                            min_invalid_count = invalid_count
                            best_centers = [(r, c, cross_pixels)]
                        elif invalid_count == min_invalid_count:
                            best_centers.append((r, c, cross_pixels))
        
        # Apply all best centers for this component
        for r, c, pixels in best_centers:
            for cr, cc in pixels:
                if 0 <= cr < H and 0 <= cc < W:
                    if out[cr, cc] == 5:
                        out[cr, cc] = 8
                        
    return out

def solve_red_cross_at_variable_scale(solver) -> Optional[List[np.ndarray]]:
    # Task 50846271
    # Check if this rule fits with L=2 or L=3 (L=3 for Example 0 specifically)
    consistent = True
    ls = []
    for i, (inp, out) in enumerate(solver.pairs):
        l_found = False
        for l in [2, 3]:
            res = solve_grid_50846271(inp, L=l)
            if np.array_equal(res, out):
                ls.append(l)
                l_found = True
                break
        if not l_found:
            consistent = False; break
            
    if consistent:
        # Heuristic for test: use L=2 unless components look like Example 0 (which had L=3)
        # Actually, let's just use L=2 as default or the most common L found.
        # Most common L:
        best_l = 2
        if ls:
            best_l = max(set(ls), key=ls.count)
        return [solve_grid_50846271(ti, L=best_l) for ti in solver.test_in]
    return None
