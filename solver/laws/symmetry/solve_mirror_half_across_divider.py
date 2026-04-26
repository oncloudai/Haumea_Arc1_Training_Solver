import numpy as np
from typing import List, Optional

def solve_mirror_half_across_divider(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a solid uniform color line (row or column) that acts as a divider.
    Mirrors one half of the grid across the divider and combines it with the other half 
    using np.maximum (preserving non-zero pixels).
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # 1. Search for vertical divider
        divider_col = -1
        for c in range(w):
            if np.all(grid[:, c] == grid[0, c]) and grid[0, c] != 0:
                divider_col = c
                break
                
        if divider_col != -1:
            left = grid[:, :divider_col]
            right = grid[:, divider_col+1:]
            # Mirror the right half horizontally
            right_mirrored = np.fliplr(right)
            
            min_w = min(left.shape[1], right_mirrored.shape[1])
            # If shapes differ significantly, this might not be it
            if abs(left.shape[1] - right_mirrored.shape[1]) > 2: pass
            
            output = np.maximum(left[:, :min_w], right_mirrored[:, :min_w])
            return output

        # 2. Search for horizontal divider
        divider_row = -1
        for r in range(h):
            if np.all(grid[r, :] == grid[r, 0]) and grid[r, 0] != 0:
                divider_row = r
                break
                
        if divider_row != -1:
            top = grid[:divider_row, :]
            bottom = grid[divider_row+1:, :]
            # Mirror the bottom half vertically
            bottom_mirrored = np.flipud(bottom)
            
            min_h = min(top.shape[0], bottom_mirrored.shape[0])
            output = np.maximum(top[:min_h, :], bottom_mirrored[:min_h, :])
            return output
            
        return None

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
