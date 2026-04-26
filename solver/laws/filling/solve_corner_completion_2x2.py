
import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_corner_completion_2x2(solver) -> Optional[List[np.ndarray]]:
    def process(grid):
        out = grid.copy()
        for color in range(1, 10):
            mask = (grid == color)
            labeled, n = label(mask, structure=np.ones((3,3)))
            for i in range(1, n+1):
                coords = np.argwhere(labeled == i)
                if len(coords) == 3:
                    min_r, min_c = coords.min(axis=0); max_r, max_c = coords.max(axis=0)
                    if max_r - min_r == 1 and max_c - min_c == 1:
                        for r in range(min_r, max_r + 1):
                            for c in range(min_c, max_c + 1):
                                if not np.any(np.all(coords == [r, c], axis=1)):
                                    out[r, c] = 1 # Color 1 is added in 3aa6fb7a
        return out

    for inp, out in solver.pairs:
        if not np.array_equal(process(inp), out): return None
    return [process(ti) for ti in solver.test_in]
