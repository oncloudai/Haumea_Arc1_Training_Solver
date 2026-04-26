import numpy as np
from typing import List, Optional

def solve_ca(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        rules = {}; consistent = True
        for inp, out in solver.pairs:
            if inp.shape != out.shape: consistent = False; break
            h, w = inp.shape; padded = np.pad(inp, 1, constant_values=bg)
            for r in range(h):
                for c in range(w):
                    window = tuple(padded[r:r+3, c:c+3].flatten())
                    if window in rules and rules[window] != out[r, c]: consistent = False; break
                    rules[window] = out[r, c]
            if not consistent: break
        if consistent and rules:
            # Stricter: must match ALL train outputs exactly
            all_match = True
            for inp, out in solver.pairs:
                h, w = inp.shape; padded = np.pad(inp, 1, constant_values=bg); pred = np.zeros_like(out)
                for r in range(h):
                    for c in range(w):
                        window = tuple(padded[r:r+3, c:c+3].flatten())
                        if window not in rules: all_match = False; break
                        pred[r, c] = rules[window]
                    if not all_match: break
                if not all_match or not np.array_equal(pred, out): all_match = False; break
            if not all_match: continue
            
            results = []
            for ti in solver.test_in:
                h, w = ti.shape; padded = np.pad(ti, 1, constant_values=bg); res = np.zeros_like(ti)
                for r in range(h):
                    for c in range(w):
                        window = tuple(padded[r:r+3, c:c+3].flatten())
                        if window not in rules: return None # Can't generalize
                        res[r, c] = rules[window]
                results.append(res)
            return results
    return None
