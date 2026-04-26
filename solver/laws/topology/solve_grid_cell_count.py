import numpy as np
from typing import List, Optional

def solve_grid_cell_count(solver) -> Optional[List[np.ndarray]]:
    def process(g):
        h_, w_ = g.shape
        # Try every color as delimiter
        for dr in range(1, 10):
            rows = np.where(np.all(g == dr, axis=1))[0]
            cols = np.where(np.all(g == dr, axis=0))[0]
            if len(rows) > 0 or len(cols) > 0:
                rb = [-1] + sorted(list(rows)) + [h_]
                cb = [-1] + sorted(list(cols)) + [w_]
                nr, nc = len(rb)-1, len(cb)-1
                # Find most frequent non-delimiter color
                unq, cnts = np.unique(g, return_counts=True)
                # Filter out bg (0) and delimiter
                others = [(c, cnt) for c, cnt in zip(unq, cnts) if c != dr and c != 0]
                fc = max(others, key=lambda x: x[1])[0] if others else 0
                return np.full((nr, nc), fc)
        return None

    consistent = True; results = []
    for inp, out in solver.pairs:
        p = process(inp)
        if p is None or not np.array_equal(p, out): consistent = False; break
    if consistent:
        for ti in solver.test_in:
            p = process(ti)
            if p is not None: results.append(p)
            else: break
        if len(results) == len(solver.test_in): return results
    return None
