import numpy as np
from typing import List, Optional

def solve_grid_a68b268e(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    
    # Find the divider cross (color 1)
    # Find row and col that are mostly color 1
    div_row = -1
    for r in range(h):
        if np.sum(grid[r, :] == 1) > h // 2:
            div_row = r
            break
            
    div_col = -1
    for c in range(w):
        if np.sum(grid[:, c] == 1) > w // 2:
            div_col = c
            break
            
    if div_row == -1 or div_col == -1:
        # Fallback to 4, 4 if not found
        div_row, div_col = 4, 4
        
    q_h, q_w = div_row, div_col
    tl = grid[0:div_row, 0:div_col]
    
    # Ensure they have same shape for safety
    def get_quad(r1, r2, c1, c2):
        q = grid[r1:r2, c1:c2]
        res = np.zeros((q_h, q_w), dtype=int)
        qh, qw = q.shape
        res[:min(qh, q_h), :min(qw, q_w)] = q[:min(qh, q_h), :min(qw, q_w)]
        return res

    tl = get_quad(0, div_row, 0, div_col)
    tr = get_quad(0, div_row, div_col+1, w)
    bl = get_quad(div_row+1, h, 0, div_col)
    br = get_quad(div_row+1, h, div_col+1, w)
    
    output_grid = np.zeros((q_h, q_w), dtype=int)
    
    # Priority: TL > TR > BL > BR
    for r in range(q_h):
        for c in range(q_w):
            if tl[r, c] != 0:
                output_grid[r, c] = tl[r, c]
            elif tr[r, c] != 0:
                output_grid[r, c] = tr[r, c]
            elif bl[r, c] != 0:
                output_grid[r, c] = bl[r, c]
            elif br[r, c] != 0:
                output_grid[r, c] = br[r, c]
                
    return output_grid

def solve_quadrant_priority_overlap_by_cross(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_a68b268e(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_a68b268e(ti) for ti in solver.test_in]
    return None
