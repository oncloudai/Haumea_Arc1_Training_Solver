import numpy as np
from typing import List, Optional

def solve_row_uniformity_check(solver) -> Optional[List[np.ndarray]]:
    for inp, out in solver.pairs:
        pred = np.zeros_like(out)
        for r in range(inp.shape[0]):
            if len(np.unique(inp[r, :])) == 1: pred[r, :] = 5
        if not np.array_equal(pred, out): return None
    results = []
    for ti in solver.test_in:
        res = np.zeros_like(ti)
        for r in range(ti.shape[0]):
            if len(np.unique(ti[r, :])) == 1: res[r, :] = 5
        results.append(res)
    return results
