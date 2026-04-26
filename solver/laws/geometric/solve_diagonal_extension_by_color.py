
import numpy as np
from typing import List, Optional

def solve_diagonal_extension_by_color(solver) -> Optional[List[np.ndarray]]:
    """
    Extends diagonal trails from colored blocks:
    Color 1 extends toward top-left.
    Color 2 extends toward bottom-right.
    """
    def apply_logic(input_grid):
        input_grid = np.array(input_grid)
        output_grid = input_grid.copy()
        colors = np.unique(input_grid[np.nonzero(input_grid)])
        
        found_any = False
        for color in colors:
            rows, cols = np.where(input_grid == color)
            if len(rows) > 0:
                min_r, max_r = rows.min(), rows.max()
                min_c, max_c = cols.min(), cols.max()

                if color == 1:
                    r, c = min_r - 1, min_c - 1
                    while r >= 0 and c >= 0:
                        output_grid[r, c] = color
                        r -= 1; c -= 1; found_any = True
                elif color == 2:
                    r, c = max_r + 1, max_c + 1
                    while r < output_grid.shape[0] and c < output_grid.shape[1]:
                        output_grid[r, c] = color
                        r += 1; c += 1; found_any = True
                        
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
