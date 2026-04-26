import numpy as np
from typing import List, Optional

def solve_extend_row_periodicity_horizontally(solver) -> Optional[List[np.ndarray]]:
    """
    For each row in the input, find its shortest horizontal period.
    Extend this period to the output width (which is twice the input width).
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        out = np.zeros((h, 2 * w), dtype=int)
        for r in range(h):
            row = grid[r, :]
            # Find the shortest horizontal period p
            period = w
            for p in range(1, w // 2 + 1):
                # Check if row[0:w-p] matches row[p:w]
                if np.array_equal(row[0 : w - p], row[p : w]):
                    period = p
                    break
            
            # Use the period to fill out the row
            for c in range(2 * w):
                out[r, c] = row[c % period]
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
