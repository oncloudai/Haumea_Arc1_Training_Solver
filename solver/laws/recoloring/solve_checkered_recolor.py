import numpy as np
from typing import List, Optional

def solve_checkered_recolor(solver) -> Optional[List[np.ndarray]]:
    for ph in [1, 2]:
        for pw in [1, 2]:
            consistent = True; results = []
            for inp, out in solver.pairs:
                if inp.shape != out.shape: consistent = False; break
                unq_out = np.unique(out[out != 0])
                if len(unq_out) != 2: consistent = False; break
                c1, c2 = unq_out
                worked = False
                for ca, cb in [(c1, c2), (c2, c1)]:
                    pred = inp.copy()
                    for r in range(inp.shape[0]):
                        for c in range(inp.shape[1]):
                            if inp[r, c] != 0:
                                v = (r//ph + c//pw) % 2
                                pred[r, c] = ca if v == 0 else cb
                    if np.array_equal(pred, out): worked = True; break
                if not worked: consistent = False; break
            if consistent:
                last_inp, last_out = solver.pairs[-1]
                unq_last = np.unique(last_out[last_out != 0]); ca, cb = unq_last[0], unq_last[1]
                pred_a = last_inp.copy()
                for r in range(last_inp.shape[0]):
                    for c in range(last_inp.shape[1]):
                        if last_inp[r, c] != 0:
                            v = (r//ph + c//pw) % 2
                            pred_a[r, c] = ca if v == 0 else cb
                if not np.array_equal(pred_a, last_out): ca, cb = cb, ca
                for ti in solver.test_in:
                    res = ti.copy()
                    for r in range(ti.shape[0]):
                        for c in range(ti.shape[1]):
                            if ti[r, c] != 0:
                                v = (r//ph + c//pw) % 2
                                res[r, c] = ca if v == 0 else cb
                    results.append(res)
                return results
    return None
