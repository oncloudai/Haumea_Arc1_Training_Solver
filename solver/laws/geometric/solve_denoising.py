import numpy as np
from typing import List, Optional

def solve_denoising(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        consistent = True; rem_colors = set(); found_change = False
        for inp, out in solver.pairs:
            if inp.shape != out.shape: consistent = False; break
            diff = (inp != out); rem_c = np.unique(inp[diff])
            rem_colors.update(rem_c); pred = inp.copy()
            for c in rem_c: pred[inp == c] = bg
            if not np.array_equal(pred, out): consistent = False; break
            if np.any(diff): found_change = True
        if consistent and found_change:
            results = []
            for ti in solver.test_in:
                res = ti.copy()
                for c in rem_colors: res[ti == c] = bg
                results.append(res)
            return results
    return None
