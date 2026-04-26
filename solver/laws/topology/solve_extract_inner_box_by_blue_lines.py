import numpy as np
from typing import List, Optional

def solve_grid_77fdfe62(grid):
    grid = np.array(grid)
    h, w = grid.shape
    blue_rows = [r for r in range(h) if np.sum(grid[r] == 1) > w * 0.5]
    blue_cols = [c for c in range(w) if np.sum(grid[:, c] == 1) > h * 0.5]
    if len(blue_rows) < 2 or len(blue_cols) < 2: return grid
    r_start, r_end = blue_rows[0] + 1, blue_rows[-1] - 1
    c_start, c_end = blue_cols[0] + 1, blue_cols[-1] - 1
    inner_h, inner_w = r_end - r_start + 1, c_end - c_start + 1
    inner_box = grid[r_start:r_end+1, c_start:c_end+1]
    n_h, n_w = inner_h // 2, inner_w // 2
    c_tl, c_tr = int(grid[0, 0]), int(grid[0, w-1])
    c_bl, c_br = int(grid[h-1, 0]), int(grid[h-1, w-1])
    out = np.zeros((inner_h, inner_w), dtype=int)
    for r in range(inner_h):
        for c in range(inner_w):
            if inner_box[r, c] == 8:
                if r < n_h:
                    if c < n_w: out[r, c] = c_tl
                    else: out[r, c] = c_tr
                else:
                    if c < n_w: out[r, c] = c_bl
                    else: out[r, c] = c_br
    return out

def solve_extract_inner_box_by_blue_lines(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_77fdfe62(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_77fdfe62(ti) for ti in solver.test_in]
    return None
