import numpy as np
from typing import List, Optional

def solve_pattern_extension_1d(solver) -> Optional[List[np.ndarray]]:
    bg = 0
    def process(ti):
        h, w = ti.shape; coords = np.argwhere(ti != bg)
        if len(coords) < 2: return None
        seeds = sorted([{'r': r, 'c': c, 'color': ti[r, c]} for r, c in coords], key=lambda x: (x['r'], x['c']))
        s1, s2 = seeds[0], seeds[1]; dr, dc = abs(s2['r'] - s1['r']), abs(s2['c'] - s1['c'])
        res = np.zeros_like(ti)
        if dc > 0 and (dr == 0 or dc < dr):
            P = 2 * dc
            for c in range(s1['c'], w):
                off = c - s1['c']
                if off % P == 0: res[:, c] = s1['color']
                elif off % P == dc: res[:, c] = s2['color']
        else:
            P = 2 * dr
            for r in range(s1['r'], h):
                off = r - s1['r']
                if off % P == 0: res[r, :] = s1['color']
                elif off % P == dr: res[r, :] = s2['color']
        return res
    results = []
    for ti in solver.test_in:
        p = process(ti)
        if p is None: return None
        results.append(p)
    for inp, out in solver.pairs:
        p = process(inp)
        if p is None or not np.array_equal(p, out): return None
    return results
