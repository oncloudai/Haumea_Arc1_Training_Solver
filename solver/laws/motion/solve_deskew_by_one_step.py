import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_deskew_by_one_step(solver) -> Optional[List[np.ndarray]]:
    bg = 0
    def process(grid):
        res = np.full_like(grid, bg); h, w = grid.shape
        blobs = get_blobs(grid, bg, connectivity=8)
        for b in blobs:
            row_map = {}
            for r, c in b['coords']:
                if r not in row_map: row_map[r] = []
                row_map[r].append(c)
            sorted_rows = sorted(row_map.keys(), reverse=True)
            new_row_map = {}
            bot_r = sorted_rows[0]; new_row_map[bot_r] = sorted(row_map[bot_r])
            for i in range(1, len(sorted_rows)):
                r = sorted_rows[i]; r_below = sorted_rows[i-1]
                curr_cols = sorted(row_map[r]); below_cols = new_row_map[r_below]
                c_min, c_max = curr_cols[0], curr_cols[-1]; b_min, b_max = below_cols[0], below_cols[-1]
                s_l = 1 if c_min < b_min else 0; s_r = 1 if c_max < b_max else 0
                if len(curr_cols) > 2: new_cols = [c + s_l for c in curr_cols]
                elif len(curr_cols) == 1: new_cols = [curr_cols[0] + s_l]
                else: new_cols = [curr_cols[0] + s_l, curr_cols[1] + s_r]
                new_row_map[r] = sorted(new_cols)
            for r, cols in new_row_map.items():
                for c in cols:
                    if 0 <= r < h and 0 <= c < w: res[r, c] = b['color']
        return res
    for inp, out in solver.pairs:
        if not np.array_equal(process(inp), out): return None
    return [process(ti) for ti in solver.test_in]
