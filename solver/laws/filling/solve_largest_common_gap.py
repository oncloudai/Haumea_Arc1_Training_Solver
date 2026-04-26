import numpy as np
from typing import List, Optional

def solve_largest_common_gap(solver) -> Optional[List[np.ndarray]]:
    def process(grid):
        h, w = grid.shape
        row_gaps = []
        for r in range(h):
            gaps = []; row = grid[r]; c = 0
            while c < w:
                if row[c] == 0:
                    s = c
                    while c < w and row[c] == 0: c += 1
                    gaps.append((s, c - 1))
                else: c += 1
            row_gaps.append(gaps)
        best_block, max_area = None, 0
        for r1 in range(h):
            for s1, e1 in row_gaps[r1]:
                for s in range(s1, e1):
                    for e in range(s + 1, e1 + 1):
                        rows = [r1]
                        for r_next in range(r1 + 1, h):
                            found = False
                            for s_next, e_next in row_gaps[r_next]:
                                if s_next <= s and e_next >= e: found = True; break
                            if found: rows.append(r_next)
                            else: break
                        if len(rows) >= 2:
                            area = len(rows) * (e - s + 1)
                            if area > max_area: max_area = area; best_block = (rows, s, e)
        out = grid.copy()
        if best_block:
            rows, s, e = best_block
            for r in rows: out[r, s:e+1] = 6
        return out

    for inp, out in solver.pairs:
        if not np.array_equal(process(inp), out): return None
    return [process(ti) for ti in solver.test_in]
