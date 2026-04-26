import numpy as np
from typing import List, Optional

def solve_vertical_periodic_extension(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies the range of rows [r1, r2] containing all non-zero pixels.
    Extracts this block of rows as a pattern P.
    Tiles P vertically to fill the entire grid, aligned with its original position.
    Specifically, out[r, :] = P[(r - r1) % len(P)].
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        rows = np.where(np.any(grid != 0, axis=1))[0]
        if len(rows) == 0: return None
        
        r1, r2 = rows.min(), rows.max()
        P = grid[r1 : r2 + 1, :]
        ph = P.shape[0]
        
        out = np.zeros_like(grid)
        for r in range(h):
            out[r, :] = P[(r - r1) % ph]
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
