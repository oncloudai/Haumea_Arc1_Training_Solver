
import numpy as np
from typing import List, Optional

def solve_double_reflection(solver) -> Optional[List[np.ndarray]]:
    def process(grid):
        h, w = grid.shape
        out = np.zeros((2 * h, 2 * w), dtype=int)
        out[0:h, 0:w] = grid
        out[0:h, w:2*w] = np.fliplr(grid)
        out[h:2*h, 0:2*w] = np.flipud(out[0:h, 0:2*w])
        return out

    for inp, out in solver.pairs:
        if not np.array_equal(process(inp), out): return None
    return [process(ti) for ti in solver.test_in]
