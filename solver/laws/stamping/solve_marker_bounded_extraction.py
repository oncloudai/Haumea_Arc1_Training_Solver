import numpy as np
from typing import List, Optional

def solve_marker_bounded_extraction(solver) -> Optional[List[np.ndarray]]:
    def solve_3de23699(grid):
        h, w = grid.shape
        for color in range(1, 10):
            coords = np.argwhere(grid == color)
            if len(coords) >= 4:
                rows = sorted(list(set(coords[:, 0]))); cols = sorted(list(set(coords[:, 1])))
                for i1 in range(len(rows)):
                    for i2 in range(i1 + 1, len(rows)):
                        for j1 in range(len(cols)):
                            for j2 in range(j1 + 1, len(cols)):
                                r1, r2, c1, c2 = rows[i1], rows[i2], cols[j1], cols[j2]
                                if grid[r1, c1] == color and grid[r1, c2] == color and \
                                   grid[r2, c1] == color and grid[r2, c2] == color:
                                    interior = grid[r1+1:r2, c1+1:c2]
                                    if interior.size == 0: continue
                                    uc = np.unique(interior)
                                    obj_color = -1
                                    for c in uc:
                                        if c != 0 and c != color: obj_color = c; break
                                    if obj_color != -1:
                                        extracted = interior.copy()
                                        extracted[extracted == obj_color] = color
                                        return extracted
        return None

    def solve_3f7978a0(grid):
        h, w = grid.shape
        cols_5 = [c for c in range(w) if np.any(grid[:, c] == 5)]
        if len(cols_5) >= 2:
            for i in range(len(cols_5)):
                for j in range(i + 1, len(cols_5)):
                    c1, c2 = cols_5[i], cols_5[j]
                    rows_5 = np.where((grid[:, c1] == 5) | (grid[:, c2] == 5))[0]
                    if len(rows_5) == 0: continue
                    rmin, rmax = min(rows_5), max(rows_5)
                    r_start, r_end = rmin, rmax
                    for r in range(rmin - 1, -1, -1):
                        if grid[r, c1] == 8 and grid[r, c2] == 8: r_start = r; break
                    for r in range(rmax + 1, h):
                        if grid[r, c1] == 8 and grid[r, c2] == 8: r_end = r; break
                    return grid[r_start:r_end+1, c1:c2+1].copy()
        return None

    # Try both strategies
    for strat in [solve_3de23699, solve_3f7978a0]:
        all_ok = True
        for inp, out in solver.pairs:
            pred = strat(inp)
            if pred is None or not np.array_equal(pred, out): all_ok = False; break
        if all_ok: return [strat(ti) for ti in solver.test_in]
    return None
