
import numpy as np
from typing import List, Optional

def solve_fill_rectangle_from_two_points(solver) -> Optional[List[np.ndarray]]:
    """
    Two same-colored isolated points define opposite corners of a filled rectangle.
    Fills that rectangle with the color.
    """
    def apply_logic(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        out = grid.copy()
        
        unique_colors = np.unique(grid)
        found_any = False
        for color in unique_colors:
            if color == 0: continue
            r_coords, c_coords = np.where(grid == color)
            if len(r_coords) == 2:
                r_min, r_max = r_coords.min(), r_coords.max()
                c_min, c_max = c_coords.min(), c_coords.max()
                out[r_min:r_max+1, c_min:c_max+1] = color
                found_any = True
        return out, found_any

    for inp, out in solver.pairs:
        pred, found = apply_logic(inp)
        if not found or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res, _ = apply_logic(ti)
        results.append(res)
    return results
