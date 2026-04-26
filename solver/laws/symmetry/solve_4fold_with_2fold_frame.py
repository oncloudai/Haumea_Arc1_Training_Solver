import numpy as np
from typing import List, Optional

def get_sym_coords(r, c):
    res = []
    # Main 28x28 (2..29, 2..29)
    if r >= 2 and c >= 2:
        # Diagonal d1: (r,c) -> (c,r)
        # Horizontal h: (r,c) -> (r, 31-c)
        # Vertical v: (r,c) -> (31-r, c)
        # Combined:
        res = [
            (r, 31-c), (31-r, c), (31-r, 31-c),
            (c, r), (c, 31-r), (31-c, r), (31-c, 31-r)
        ]
    elif r < 2:
        # Row i (i=0,1). Palindrome: (r, c) -> (r, 31-c). Row-Col: (r, c) -> (c, r).
        # Col-Palindrome: (c, r) -> (31-c, r)
        res = [(r, 31-c), (c, r), (31-c, r)]
    elif c < 2:
        # Col i (i=0,1). Palindrome: (r, c) -> (31-r, c). Col-Row: (r, c) -> (c, r).
        # Row-Palindrome: (c, r) -> (c, 31-r)
        res = [(31-r, c), (c, r), (r, 31-c)]
    return res

def solve_4fold_with_2fold_frame(solver) -> Optional[List[np.ndarray]]:
    def apply(grid):
        out = grid.copy()
        for _ in range(5):
            changed = False
            for r in range(30):
                for c in range(30):
                    if out[r, c] == 9:
                        syms = get_sym_coords(r, c)
                        for sr, sc in syms:
                            if 0 <= sr < 30 and 0 <= sc < 30:
                                val = out[sr, sc]
                                if val != 9:
                                    out[r, c] = val
                                    changed = True; break
            if not changed: break
        return out

    # Verify on training pairs
    for inp, out_expected in solver.pairs:
        if inp.shape != (30, 30): return None
        pred = apply(inp)
        if not np.array_equal(pred, out_expected):
            return None
            
    # Apply to test
    results = []
    for ti in solver.test_in:
        if ti.shape != (30, 30): return None
        res = apply(ti)
        results.append(res)
    return results
