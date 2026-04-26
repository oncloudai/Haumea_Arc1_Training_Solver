import numpy as np
from typing import List, Optional

def solve_tiled_checkerboard(solver) -> Optional[List[np.ndarray]]:
    for color8 in range(1, 10):
        for cond in [lambda r,c: (r+c)%2==0, lambda r,c: (r+c)%2==1, 
                     lambda r,c: r%2==0, lambda r,c: r%2==1,
                     lambda r,c: c%2==0, lambda r,c: c%2==1]:
            consistent = True; found_any = False
            for inp, out in solver.pairs:
                h, w = inp.shape; tiled = np.tile(inp, (2, 2))
                if out.shape != tiled.shape: consistent = False; break
                pred = tiled.copy()
                for r in range(pred.shape[0]):
                    for c in range(pred.shape[1]):
                        if pred[r, c] == 0 and cond(r, c): pred[r, c] = color8; found_any = True
                if not np.array_equal(pred, out): consistent = False; break
            if consistent and found_any:
                results = []
                for ti in solver.test_in:
                    h, w = ti.shape; res = np.tile(ti, (2, 2))
                    for r in range(res.shape[0]):
                        for c in range(res.shape[1]):
                            if res[r, c] == 0 and cond(r, c): res[r, c] = color8
                    results.append(res)
                return results
    return None
