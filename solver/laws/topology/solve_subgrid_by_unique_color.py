import numpy as np
from typing import List, Optional

def get_all_rects(grid):
    h, w = grid.shape
    u_rows = [r for r in range(h) if len(np.unique(grid[r])) == 1]
    u_cols = [c for c in range(w) if len(np.unique(grid[:, c])) == 1]
    
    def get_parts(total, indices):
        parts = []; last = -1
        for idx in sorted(indices):
            if idx > last + 1: parts.append((last + 1, idx))
            last = idx
        if last < total - 1: parts.append((last + 1, total))
        return parts

    r_parts = get_parts(h, u_rows); c_parts = get_parts(w, u_cols)
    rects = []
    for r1, r2 in r_parts:
        for c1, c2 in c_parts:
            rects.append(grid[r1:r2, c1:c2])
    return rects

def solve_subgrid_by_unique_color(solver) -> Optional[List[np.ndarray]]:
    for inp, out in solver.pairs:
        rects = get_all_rects(inp)
        unique_color_rects = []
        for i, r1 in enumerate(rects):
            colors1 = set(np.unique(r1[r1 != 0]))
            if not colors1: continue
            is_unique = False
            for c in colors1:
                appeared_elsewhere = False
                for j, r2 in enumerate(rects):
                    if i == j: continue
                    if c in set(np.unique(r2)):
                        appeared_elsewhere = True; break
                if not appeared_elsewhere: is_unique = True; break
            if is_unique: unique_color_rects.append(r1)
        if len(unique_color_rects) != 1 or not np.array_equal(unique_color_rects[0], out):
            return None
            
    results = []
    for ti in solver.test_in:
        rects = get_all_rects(ti)
        matches = []
        for i, r1 in enumerate(rects):
            colors1 = set(np.unique(r1[r1 != 0]))
            if not colors1: continue
            has_unique = False
            for c in colors1:
                elsewhere = False
                for j, r2 in enumerate(rects):
                    if i == j: continue
                    if c in set(np.unique(r2)): elsewhere = True; break
                if not elsewhere: has_unique = True; break
            if has_unique: matches.append(r1)
        if len(matches) == 1: results.append(matches[0])
        else: return None
    return results
