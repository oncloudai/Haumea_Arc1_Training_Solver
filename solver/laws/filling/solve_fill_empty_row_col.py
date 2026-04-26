import numpy as np
from typing import List, Optional

def solve_fill_empty_row_col(solver) -> Optional[List[np.ndarray]]:
    # Try to find the fill color from training pairs
    fill_color = None
    for inp, out in solver.pairs:
        diff = (inp != out)
        if np.any(diff):
            fill_colors = np.unique(out[diff])
            if len(fill_colors) == 1:
                if fill_color is None: fill_color = fill_colors[0]
                elif fill_color != fill_colors[0]: return None # Inconsistent fill color
            else: return None
            
    if fill_color is None: return None
    
    # Verify the rule on all pairs
    for inp, out in solver.pairs:
        pred = inp.copy()
        h, w = inp.shape
        for r in range(h):
            if np.all(inp[r, :] == 0): pred[r, :] = fill_color
        for c in range(w):
            if np.all(inp[:, c] == 0): pred[:, c] = fill_color
        if not np.array_equal(pred, out): return None
        
    # Apply to test inputs
    results = []
    for ti in solver.test_in:
        res = ti.copy()
        h, w = ti.shape
        for r in range(h):
            if np.all(ti[r, :] == 0): res[r, :] = fill_color
        for c in range(w):
            if np.all(ti[:, c] == 0): res[:, c] = fill_color
        results.append(res)
    return results
