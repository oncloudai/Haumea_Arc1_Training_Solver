import numpy as np
from typing import List, Optional

def solve_repeat_pixel_and_add_5s(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a single non-zero pixel at (r, c) with color X.
    Repeats color X in every row, in columns c, c+2, c+4...
    Adds color 5 in row 0 at columns c+1, c+5, c+9...
    Adds color 5 in the last row at columns c+3, c+7, c+11...
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        rows, cols = np.where(grid != 0)
        if len(rows) != 1: return None
        
        r, c = rows[0], cols[0]
        color = grid[r, c]
        
        out = np.zeros_like(grid)
        # Repeat main color
        for row in range(h):
            for col in range(c, w, 2):
                out[row, col] = color
                
        # Add 5s
        for col in range(c + 1, w, 4):
            out[0, col] = 5
        for col in range(c + 3, w, 4):
            out[h - 1, col] = 5
            
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
