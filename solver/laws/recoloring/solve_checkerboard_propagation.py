
import numpy as np
from typing import List, Optional

def solve_checkerboard_propagation(solver) -> Optional[List[np.ndarray]]:
    def process(grid):
        out = grid.copy(); h, w = grid.shape
        for c in range(w):
            color = grid[0, c]
            if color != 0:
                for r in range(1, h):
                    if r % 2 == 0: out[r, c] = color
                    else:
                        if c - 1 >= 0: out[r, c - 1] = color
                        if c + 1 < w: out[r, c + 1] = color
        return out

    for inp, out in solver.pairs:
        if not np.array_equal(process(inp), out): return None
    return [process(ti) for ti in solver.test_in]
