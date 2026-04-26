import numpy as np
from typing import List, Optional

def solve_ray_symmetry_fixed(solver) -> Optional[List[np.ndarray]]:
    # In 3bd67248, colors are 2 and 4.
    c1, c2 = 2, 4
    for inp, out in solver.pairs:
        h, w = inp.shape; pred = inp.copy()
        for i in range(h - 1):
            c = w - 1 - i
            if 0 <= c < w: pred[i, c] = c1
        if h > 0: pred[h-1, 1:] = c2
        if not np.array_equal(pred, out): return None
    results = []
    for ti in solver.test_in:
        h, w = ti.shape; res = ti.copy()
        for i in range(h - 1):
            c = w - 1 - i
            if 0 <= c < w: res[i, c] = c1
        if h > 0: res[h-1, 1:] = c2
        results.append(res)
    return results
