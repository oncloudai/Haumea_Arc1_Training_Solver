import numpy as np
from typing import List, Optional

def solve_diagonal_down_right_propagation(solver) -> Optional[List[np.ndarray]]:
    consistent = True; found_any = False
    for inp, out in solver.pairs:
        hi, wi = inp.shape
        ho, wo = out.shape
        pred = np.zeros_like(out)
        for r in range(hi):
            for c in range(wi):
                if inp[r, c] != 0:
                    color = inp[r, c]
                    # Propagate down-right
                    for k in range(max(ho, wo)):
                        nr, nc = r + k, c + k
                        if 0 <= nr < ho and 0 <= nc < wo:
                            # If there's already a color, which one wins?
                            # Usually later ones in the input loop (lower rows/cols)
                            pred[nr, nc] = color
                            found_any = True
        
        if not np.array_equal(pred, out):
            consistent = False; break
            
    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            hi, wi = ti.shape
            # Infer output shape from training
            ho, wo = solver.train_out[0].shape
            res = np.zeros((ho, wo), dtype=int)
            for r in range(hi):
                for c in range(wi):
                    if ti[r, c] != 0:
                        for k in range(max(ho, wo)):
                            nr, nc = r + k, c + k
                            if 0 <= nr < ho and 0 <= nc < wo:
                                res[nr, nc] = ti[r, c]
            results.append(res)
        return results
    return None
