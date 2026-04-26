import numpy as np
from typing import List, Optional

def solve_pixel_upscaling_by_factor(solver) -> Optional[List[np.ndarray]]:
    for f in range(2, 6):
        consistent = True
        for inp, out in solver.pairs:
            h, w = inp.shape
            if out.shape != (h * f, w * f):
                consistent = False; break
            
            # Upscale inp by factor f
            pred = np.zeros_like(out)
            for r in range(h):
                for c in range(w):
                    pred[r*f:(r+1)*f, c*f:(c+1)*f] = inp[r, c]
            
            if not np.array_equal(pred, out):
                consistent = False; break
        
        if consistent:
            results = []
            for ti in solver.test_in:
                h, w = ti.shape
                res = np.zeros((h * f, w * f), dtype=int)
                for r in range(h):
                    for c in range(w):
                        res[r*f:(r+1)*f, c*f:(c+1)*f] = ti[r, c]
                results.append(res)
            return results
    return None
