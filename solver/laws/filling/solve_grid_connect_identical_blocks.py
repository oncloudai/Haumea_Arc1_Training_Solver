import numpy as np
from typing import List, Optional

def solve_grid_connect_identical_blocks(solver) -> Optional[List[np.ndarray]]:
    bg = 0
    def find_delim(grid):
        h, w = grid.shape
        for d in range(1, 10):
            rows = np.where(np.all(grid == d, axis=1))[0]
            cols = np.where(np.all(grid == d, axis=0))[0]
            if len(rows) >= 2 and len(cols) >= 2: return d
        return None

    def process(grid):
        delim = find_delim(grid)
        if delim is None: return None
        res = grid.copy(); h, w = grid.shape
        rows = [-1] + sorted(list(np.where(np.all(grid == delim, axis=1))[0])) + [h]
        cols = [-1] + sorted(list(np.where(np.all(grid == delim, axis=0))[0])) + [w]
        cells = []
        for r_idx in range(len(rows)-1):
            for c_idx in range(len(cols)-1):
                r1, r2 = rows[r_idx]+1, rows[r_idx+1]; c1, c2 = cols[c_idx]+1, cols[c_idx+1]
                if r2 > r1 and c2 > c1:
                    sub = grid[r1:r2, c1:c2]; unq = np.unique(sub); colors = unq[(unq != bg) & (unq != delim)]
                    if len(colors) == 1: cells.append({'r':r_idx, 'c':c_idx, 'r1':r1, 'r2':r2, 'c1':c1, 'c2':c2, 'color':colors[0]})
        for i in range(len(cells)):
            for j in range(i+1, len(cells)):
                c1, c2 = cells[i], cells[j]
                if c1['color'] == c2['color']:
                    if c1['r'] == c2['r']:
                        for k in range(min(c1['c'], c2['c']), max(c1['c'], c2['c'])+1):
                            target_sub = res[c1['r1']:c1['r2'], cols[k]+1:cols[k+1]]
                            target_sub[target_sub == bg] = c1['color']
                    elif c1['c'] == c2['c']:
                        for k in range(min(c1['r'], c2['r']), max(c1['r'], c2['r'])+1):
                            target_sub = res[rows[k]+1:rows[k+1], c1['c1']:c1['c2']]
                            target_sub[target_sub == bg] = c1['color']
        return res

    consistent = True; found_any = False
    for inp, out in solver.pairs:
        p = process(inp)
        if p is None or not np.array_equal(p, out): consistent = False; break
        if not np.array_equal(p, inp): found_any = True
    if consistent and found_any:
        return [process(ti) for ti in solver.test_in]
    return None
