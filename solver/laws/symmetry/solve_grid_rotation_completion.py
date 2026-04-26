import numpy as np
from typing import List, Optional

def solve_grid_rotation_completion(solver) -> Optional[List[np.ndarray]]:
    for ops in [[np.rot90], [np.rot90, lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 3)]]:
        consistent = True; found_change = False
        for inp, out in solver.pairs:
            res = inp.copy()
            for op in ops:
                try:
                    t = op(inp)
                    if t.shape != res.shape: consistent = False; break
                    res = np.maximum(res, t)
                except: consistent = False; break
            if not consistent or not np.array_equal(res, out): consistent = False; break
            if not np.array_equal(res, inp): found_change = True
        if consistent and found_change:
            results = []
            for ti in solver.test_in:
                res = ti.copy()
                for op in ops: res = np.maximum(res, op(ti))
                results.append(res)
            return results
    return None
