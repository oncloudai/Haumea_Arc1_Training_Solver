import numpy as np
from typing import List, Optional

def solve_periodic_recolor_3x3(solver) -> Optional[List[np.ndarray]]:
    # Strategy: Find a global 3x3 (or small NxM) recoloring pattern
    for ph in range(1, 4):
        for pw in range(1, 4):
            # Try all possible pattern contents
            # Actually, let's derive the pattern from the first pair
            inp0, out0 = solver.pairs[0]
            h0, w0 = inp0.shape
            
            # For each (r, c), check how it depends on (r % ph, c % pw)
            pattern = {} # (r%ph, c%pw, val_in) -> val_out
            consistent = True
            for r in range(h0):
                for c in range(w0):
                    key = (r % ph, c % pw, inp0[r, c])
                    if key in pattern and pattern[key] != out0[r, c]:
                        consistent = False; break
                    pattern[key] = out0[r, c]
                if not consistent: break
            
            if consistent:
                # Verify on all pairs
                all_match = True
                for inp, out in solver.pairs:
                    h, w = inp.shape
                    pred = np.zeros_like(inp)
                    for r in range(h):
                        for c in range(w):
                            key = (r % ph, c % pw, inp[r, c])
                            if key in pattern: pred[r, c] = pattern[key]
                            else: pred[r, c] = inp[r, c] # Fallback
                    if not np.array_equal(pred, out):
                        all_match = False; break
                
                if all_match:
                    results = []
                    for ti in solver.test_in:
                        h, w = ti.shape
                        res = np.zeros_like(ti)
                        for r in range(h):
                            for c in range(w):
                                key = (r % ph, c % pw, ti[r, c])
                                if key in pattern: res[r, c] = pattern[key]
                                else: res[r, c] = ti[r, c]
                        results.append(res)
                    return results
    return None
