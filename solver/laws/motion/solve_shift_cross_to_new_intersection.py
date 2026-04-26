import numpy as np
from typing import List, Optional

def solve_shift_cross_to_new_intersection(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a 'cross' structure (a row and column of the same color).
    Identifies its intersection.
    Counts pixels of color 5.
    Shifts the intersection by (+N, -N) where N is the count of color 5.
    Re-draws the cross at the new intersection.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # 1. Count color 5 pixels
        n = np.sum(grid == 5)
        
        # 2. Find the cross color and its intersection
        # The cross color is any color that isn't 0 or 5.
        cross_color = -1
        for color in range(1, 10):
            if color != 5 and np.any(grid == color):
                cross_color = color
                break
        if cross_color == -1: return None
            
        row_counts = [np.sum(grid[r, :] == cross_color) for r in range(h)]
        col_counts = [np.sum(grid[:, c] == cross_color) for c in range(w)]
        
        r_idx = np.argmax(row_counts)
        c_idx = np.argmax(col_counts)
        
        # 3. Apply shift (+N, -N)
        new_r = r_idx + n
        new_c = c_idx - n
        
        # 4. Create output grid
        out = np.zeros_like(grid)
        if 0 <= new_r < h: out[new_r, :] = cross_color
        if 0 <= new_c < w: out[:, new_c] = cross_color
        return out

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
