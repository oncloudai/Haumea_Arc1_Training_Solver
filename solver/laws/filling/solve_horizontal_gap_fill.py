import numpy as np
from typing import List, Optional

def solve_horizontal_gap_fill(solver) -> Optional[List[np.ndarray]]:
    for inp, out in solver.pairs:
        res = inp.copy()
        for r in range(inp.shape[0]):
            for c in range(1, 10):
                cols = np.where(inp[r,:] == c)[0]
                if len(cols) >= 2: res[r, cols.min():cols.max()+1] = c
        if not np.array_equal(res, out): return None
    results = []
    for ti in solver.test_in:
        res = ti.copy()
        for r in range(ti.shape[0]):
            for c in range(1, 10):
                cols = np.where(ti[r,:] == c)[0]
                if len(cols) >= 2: res[r, cols.min():cols.max()+1] = c
        results.append(res)
    return results
