import numpy as np
from typing import List, Optional

def solve_grid_7837ac64(grid):
    grid = np.array(grid)
    h, w = grid.shape
    counts = np.bincount(grid.flatten())
    sorted_colors = np.argsort(counts)[::-1]
    
    for grid_color in sorted_colors:
        if grid_color == 0: continue
        row_is_grid = [np.sum(grid[r] == grid_color) > w * 0.4 for r in range(h)]
        col_is_grid = [np.sum(grid[:, c] == grid_color) > h * 0.4 for c in range(w)]
        if sum(row_is_grid) < 2 or sum(col_is_grid) < 2: continue
        rows_idx = [i for i, x in enumerate(row_is_grid) if x]
        cols_idx = [i for i, x in enumerate(col_is_grid) if x]
        m_h, m_w = len(rows_idx), len(cols_idx)
        m = np.zeros((m_h, m_w), dtype=int)
        for i, r in enumerate(rows_idx):
            for j, c in enumerate(cols_idx):
                val = grid[r, c]
                if val != grid_color: m[i, j] = int(val)
        non_zero = np.argwhere(m > 0)
        if non_zero.size == 0: continue
        r_min, c_min = non_zero.min(axis=0)
        output = np.zeros((3, 3), dtype=int)
        for r in range(3):
            for c in range(3):
                coords = [(r_min + r, c_min + c), (r_min + r + 1, c_min + c),
                          (r_min + r, c_min + c + 1), (r_min + r + 1, c_min + c + 1)]
                colors_found = []
                for rr, cc in coords:
                    if 0 <= rr < m_h and 0 <= cc < m_w: colors_found.append(m[rr, cc])
                    else: colors_found.append(0)
                if len(set(colors_found)) == 1 and colors_found[0] > 0:
                    output[r, c] = colors_found[0]
        if np.any(output > 0): return output
    return np.zeros((3, 3), dtype=int)

def solve_grid_intersection_quad_to_3x3(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_7837ac64(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_7837ac64(ti) for ti in solver.test_in]
    return None
