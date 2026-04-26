import numpy as np
from typing import List, Optional

def solve_template_kronecker(solver) -> Optional[List[np.ndarray]]:
    def construct(p, tm):
        ph, pw = p.shape
        th, tw = tm.shape
        res = np.zeros((ph * th, pw * tw), dtype=int)
        for r in range(ph):
            for c in range(pw):
                color = p[r, c]
                if color != 0:
                    # Apply template mask
                    res[r*th:(r+1)*th, c*tw:(c+1)*tw][tm] = color
        return res

    def find_split_and_construct(grid):
        h, w = grid.shape
        # Try horizontal split
        if w % 2 == 0:
            w2 = w // 2
            h1, h2 = grid[:, :w2], grid[:, w2:]
            for p, t in [(h1, h2), (h2, h1)]:
                unq = np.unique(t); unq = unq[unq != 0]
                if len(unq) == 1:
                    return construct(p, t == unq[0])
        # Try vertical split
        if h % 2 == 0:
            h2 = h // 2
            v1, v2 = grid[:h2, :], grid[h2:, :]
            for p, t in [(v1, v2), (v2, v1)]:
                unq = np.unique(t); unq = unq[unq != 0]
                if len(unq) == 1:
                    return construct(p, t == unq[0])
        return None

    consistent = True
    for inp, out in solver.pairs:
        pred = find_split_and_construct(inp)
        if pred is None or not np.array_equal(pred, out):
            consistent = False; break
    
    if consistent:
        results = []
        for ti in solver.test_in:
            res = find_split_and_construct(ti)
            if res is None: return None
            results.append(res)
        return results
    return None
