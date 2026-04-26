
import numpy as np
from typing import List, Optional

def solve_recolor_gray_by_red_presence(solver) -> Optional[List[np.ndarray]]:
    """
    Recolors gray (5) pixels to azure (8) if they are strictly between 
    two red (2) pixels in the same row or column, with only gray (5) pixels in between.
    """
    def run_single(input_grid):
        grid = np.array(input_grid)
        out = grid.copy()
        rows, cols = grid.shape
        
        # Row-wise
        for r in range(rows):
            red_indices = np.where(grid[r] == 2)[0]
            if len(red_indices) >= 2:
                for i in range(len(red_indices) - 1):
                    c1, c2 = red_indices[i], red_indices[i+1]
                    # Check if ALL pixels between c1 and c2 are Gray (5)
                    if all(grid[r, x] == 5 for x in range(c1 + 1, c2)):
                        for x in range(c1 + 1, c2):
                            out[r, x] = 8
                            
        # Col-wise
        for c in range(cols):
            red_indices = np.where(grid[:, c] == 2)[0]
            if len(red_indices) >= 2:
                for i in range(len(red_indices) - 1):
                    r1, r2 = red_indices[i], red_indices[i+1]
                    # Check if ALL pixels between r1 and r2 are Gray (5)
                    if all(grid[x, c] == 5 for x in range(r1 + 1, r2)):
                        for x in range(r1 + 1, r2):
                            out[x, c] = 8
                            
        return out

    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if not np.array_equal(pred, out_expected):
            return None
            
    return [run_single(ti) for ti in solver.test_in]
