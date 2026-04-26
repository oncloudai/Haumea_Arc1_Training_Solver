
import numpy as np
from typing import List, Optional

def solve_meta_grid_template_stamping(solver) -> Optional[List[np.ndarray]]:
    def detect_grid(grid):
        h, w = grid.shape
        for color in range(1, 10):
            rows = np.where(np.all(grid == color, axis=1))[0]
            cols = np.where(np.all(grid == color, axis=0))[0]
            if len(rows) > 0 and len(cols) > 0:
                row_diffs = np.diff(rows)
                col_diffs = np.diff(cols)
                if len(row_diffs) > 0 and np.all(row_diffs == row_diffs[0]):
                    if len(col_diffs) > 0 and np.all(col_diffs == col_diffs[0]):
                        return rows, cols
        return None, None

    def get_meta_grid_colors(grid, rows, cols):
        r_starts = [0] + [r + 1 for r in rows]
        r_ends = [r for r in rows] + [grid.shape[0]]
        c_starts = [0] + [c + 1 for c in cols]
        c_ends = [c for c in cols] + [grid.shape[1]]
        R, C = len(r_starts), len(c_starts)
        colors = np.zeros((R, C), dtype=int)
        for r in range(R):
            for c in range(C):
                sub = grid[r_starts[r]:r_ends[r], c_starts[c]:c_ends[c]]
                unique = np.unique(sub)
                if len(unique) == 1: colors[r, c] = unique[0]
                else:
                    nz = unique[unique != 0]
                    if len(nz) > 0: colors[r, c] = nz[0]
        return colors, (r_starts, r_ends, c_starts, c_ends)

    def bfs_label(colors):
        R, C = colors.shape
        labeled = np.zeros((R, C), dtype=int)
        n = 0
        for r in range(R):
            for c in range(C):
                if colors[r,c] != 0 and labeled[r,c] == 0:
                    n += 1; q = [(r, c)]; labeled[r,c] = n
                    while q:
                        curr_r, curr_c = q.pop(0)
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < R and 0 <= nc < C and colors[nr,nc] != 0 and labeled[nr,nc] == 0:
                                labeled[nr,nc] = n; q.append((nr, nc))
        return labeled, n

    def apply_stamping(colors):
        R, C = colors.shape
        labeled, n = bfs_label(colors)
        templates = []
        for i in range(1, n + 1):
            coords = np.argwhere(labeled == i)
            if len(coords) > 1: templates.append(coords)
        out_colors = colors.copy()
        for template_coords in templates:
            for tr, tc in template_coords:
                anchor_color = colors[tr, tc]
                pattern = [(cr - tr, cc - tc, colors[cr, cc]) for cr, cc in template_coords]
                for i in range(1, n + 1):
                    coords = np.argwhere(labeled == i)
                    if len(coords) == 1:
                        r, c = coords[0]
                        if colors[r, c] == anchor_color:
                            for dr, dc, color in pattern:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < R and 0 <= nc < C: out_colors[nr, nc] = color
        return out_colors

    def process(grid):
        rows, cols = detect_grid(grid)
        if rows is None: return None
        colors, (r_starts, r_ends, c_starts, c_ends) = get_meta_grid_colors(grid, rows, cols)
        out_colors = apply_stamping(colors)
        res = grid.copy()
        for r in range(len(r_starts)):
            for c in range(len(c_starts)):
                if out_colors[r, c] != 0:
                    res[r_starts[r]:r_ends[r], c_starts[c]:c_ends[c]] = out_colors[r, c]
        return res

    for inp, out in solver.pairs:
        pred = process(inp)
        if pred is None or not np.array_equal(pred, out): return None
    return [process(ti) for ti in solver.test_in]
