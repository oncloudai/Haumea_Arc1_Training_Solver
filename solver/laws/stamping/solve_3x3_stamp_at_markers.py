import numpy as np
from typing import List, Optional

def solve_3x3_stamp_at_markers(solver) -> Optional[List[np.ndarray]]:
    """
    Stamps the top-left 3x3 block onto every position where a marker (color 1) is found,
    centering the 3x3 block on the marker.
    """
    def apply(grid):
        out = grid.copy()
        block = grid[0:3, 0:3]
        markers = np.argwhere(grid == 1)
        for mr, mc in markers:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = mr + dr, mc + dc
                    if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                        out[nr, nc] = block[dr + 1, dc + 1]
        return out

    for inp, out_expected in solver.pairs:
        if inp.shape[0] < 3 or inp.shape[1] < 3: return None
        pred = apply(inp)
        if not np.array_equal(pred, out_expected):
            return None
            
    results = []
    for ti in solver.test_in:
        results.append(apply(ti))
    return results
