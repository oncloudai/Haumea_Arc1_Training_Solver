import numpy as np
from typing import List, Optional

def solve_draw_cross_at_midpoint_of_color_1(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies the midpoint between two blue (color 1) pixels.
    Draws a green (color 3) cross centered at that midpoint.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        coords = np.argwhere(grid == 1)
        if len(coords) < 2: return grid
        
        r_mid = (coords[0][0] + coords[1][0]) // 2
        c_mid = (coords[0][1] + coords[1][1]) // 2
        
        output = grid.copy()
        cross_points = [(r_mid, c_mid), (r_mid-1, c_mid), (r_mid+1, c_mid), (r_mid, c_mid-1), (r_mid, c_mid+1)]
        for r, c in cross_points:
            if 0 <= r < output.shape[0] and 0 <= c < output.shape[1]:
                output[r, c] = 3
        return output

    for inp, out in solver.pairs:
        if not np.array_equal(apply_logic(inp), out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
