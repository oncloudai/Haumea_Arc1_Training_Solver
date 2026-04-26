import numpy as np
from typing import List, Optional

def solve_extend_horizontally_with_5(solver) -> Optional[List[np.ndarray]]:
    """
    For each non-zero pixel at (r, c) with color X:
    Extend it horizontally to the right edge.
    The pattern is X, 5, X, 5, X, 5...
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        out = grid.copy()
        
        # We assume each row has at most one pixel to start from
        # If there are multiple, we'll process them in order
        for r in range(h):
            for c in range(w):
                color = grid[r, c]
                if color != 0 and color != 5:
                    # Extend from here
                    for nc in range(c + 1, w):
                        if (nc - c) % 2 == 1:
                            out[r, nc] = 5
                        else:
                            out[r, nc] = color
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
