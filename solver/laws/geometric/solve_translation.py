import numpy as np
from typing import List, Optional

def solve_translation(solver) -> Optional[List[np.ndarray]]:
    for dr in range(-5, 6):
        for dc in range(-5, 6):
            if dr == 0 and dc == 0: continue
            mapping = {}; consistent = True; found_change = False
            for inp, out in solver.pairs:
                if inp.shape != out.shape: consistent = False; break
                h, w = inp.shape; pred = np.zeros_like(out)
                for r in range(h):
                    for c in range(w):
                        sr, sc = r - dr, c - dc
                        val_in = inp[sr, sc] if (0 <= sr < h and 0 <= sc < w) else 0
                        val_out = out[r, c]
                        if val_in in mapping and mapping[val_in] != val_out: consistent = False; break
                        mapping[val_in] = val_out
                    if not consistent: break
                if not consistent: break
                for r in range(h):
                    for c in range(w):
                        sr, sc = r - dr, c - dc
                        val_in = inp[sr, sc] if (0 <= sr < h and 0 <= sc < w) else 0
                        pred[r, c] = mapping.get(val_in, 0)
                if not np.array_equal(pred, out): consistent = False; break
                if not np.array_equal(pred, inp): found_change = True
            if consistent and found_change:
                results = []
                for ti in solver.test_in:
                    h, w = ti.shape; res = np.zeros_like(ti)
                    for r in range(h):
                        for c in range(w):
                            sr, sc = r - dr, c - dc
                            val_in = ti[sr, sc] if (0 <= sr < h and 0 <= sc < w) else 0
                            res[r, c] = mapping.get(val_in, 0)
                    results.append(res)
                return results
    return None
