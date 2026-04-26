import numpy as np
from typing import List, Optional

def solve_column_gravity_fill(solver) -> Optional[List[np.ndarray]]:
    consistent = True; found_any = False
    for inp, out in solver.pairs:
        h, w = inp.shape
        pred = inp.copy()
        for c in range(w):
            for r in range(h):
                if inp[r, c] != 0:
                    color = inp[r, c]
                    pred[r:, c] = color
                    found_any = True
        if not np.array_equal(pred, out):
            consistent = False; break
            
    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            h, w = ti.shape
            res = ti.copy()
            for c in range(w):
                for r in range(h):
                    if ti[r, c] != 0:
                        res[r:, c] = ti[r, c]
            results.append(res)
        return results
    return None
