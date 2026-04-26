import numpy as np
from typing import List, Optional

def solve_draw_diagonals_from_pixel_0(solver) -> Optional[List[np.ndarray]]:
    """
    Finds the position of a pixel with color 0.
    Sets all pixels on its diagonals (both directions) to color 0.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        out = grid.copy()
        
        coords = np.argwhere(grid == 0)
        if len(coords) == 0: return None
        
        r0, c0 = coords[0]
        for r in range(rows):
            for c in range(cols):
                if (r - c == r0 - c0) or (r + c == r0 + c0):
                    out[r, c] = 0
        return out

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
