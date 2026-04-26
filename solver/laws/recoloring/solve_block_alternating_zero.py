
import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_block_alternating_zero(solver) -> Optional[List[np.ndarray]]:
    def process(grid):
        out = grid.copy()
        for color in range(1, 10):
            mask = (grid == color)
            labeled, n = label(mask)
            for i in range(1, n + 1):
                coords = np.argwhere(labeled == i)
                min_r, min_c = coords.min(axis=0); max_r, max_c = coords.max(axis=0)
                if len(coords) == (max_r - min_r + 1) * (max_c - min_c + 1):
                    rect_h = max_r - min_r + 1
                    if rect_h >= 3:
                        mid_r = min_r + rect_h // 2
                        for c in range(min_c + 1, max_c, 2): out[mid_r, c] = 0
        return out

    for inp, out in solver.pairs:
        if not np.array_equal(process(inp), out): return None
    return [process(ti) for ti in solver.test_in]
