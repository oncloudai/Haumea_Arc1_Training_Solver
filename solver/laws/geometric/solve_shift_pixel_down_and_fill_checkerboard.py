import numpy as np
from typing import List, Optional

def solve_shift_pixel_down_and_fill_checkerboard(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a single non-zero pixel at (r, c) with color X.
    Shifts it down to (r+1, c).
    Fills all rows from 0 to r with color 4 in columns matching the parity of c.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        rows, cols = np.where(grid != 0)
        if len(rows) != 1: return None
        
        r, c = rows[0], cols[0]
        color = grid[r, c]
        
        if r + 1 >= h: return None
        
        out = np.zeros_like(grid)
        out[r + 1, c] = color
        
        for row in range(r + 1):
            for col in range(w):
                if col % 2 == c % 2:
                    out[row, col] = 4
        return out

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
