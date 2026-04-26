import numpy as np
from typing import List, Optional

def solve_fill_towards_edge_or_divider(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a central horizontal divider (color 5).
    For each pixel of color 1, it fills the vertical line between the pixel 
    and the nearest outer edge (row 0 or row H-1).
    For each pixel of color 2, it fills the vertical line between the pixel 
    and the divider (color 5 row).
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        out = grid.copy()
        
        rows_5 = np.where(grid == 5)[0]
        if len(rows_5) == 0: return None
        rd = rows_5[0]
        
        for r in range(h):
            for c in range(w):
                color = grid[r, c]
                if color == 1:
                    # Color 1 extends to outer edge
                    if r < rd:
                        out[0:r+1, c] = 1
                    else:
                        out[r:h, c] = 1
                elif color == 2:
                    # Color 2 extends to divider
                    if r < rd:
                        out[r:rd, c] = 2
                    else:
                        out[rd+1:r+1, c] = 2
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
