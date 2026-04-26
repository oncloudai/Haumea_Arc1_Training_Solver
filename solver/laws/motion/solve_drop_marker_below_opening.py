
import numpy as np
from typing import List, Optional

def solve_drop_marker_below_opening(solver) -> Optional[List[np.ndarray]]:
    """
    Find U-shaped brackets (3 cells wide, open at bottom) and place a 4
    at the bottom row of the grid, directly below the gap.
    The bracket is:
    [C, C, C]
    [C, 0, C]
    """
    def apply_logic(input_grid):
        input_grid = np.array(input_grid)
        rows, cols = input_grid.shape
        output_grid = input_grid.copy()
        found_any = False
        
        for r in range(rows - 1):
            for c in range(cols - 2):
                c1 = input_grid[r, c]
                c2 = input_grid[r, c+1]
                c3 = input_grid[r, c+2]
                c4 = input_grid[r+1, c]
                c5 = input_grid[r+1, c+1]
                c6 = input_grid[r+1, c+2]
                
                if c1 != 0 and c1 == c2 == c3 == c4 == c6 and c5 == 0:
                    # Found the bracket. Place 4 at the bottom.
                    output_grid[rows-1, c+1] = 4
                    found_any = True
        return output_grid, found_any

    for inp, out in solver.pairs:
        pred, found = apply_logic(inp)
        if not found or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res, _ = apply_logic(ti)
        results.append(res)
    return results
