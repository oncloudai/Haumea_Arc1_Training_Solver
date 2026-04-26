import numpy as np
from typing import List, Optional

def solve_recolor_vertical_segments_half_split(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies contiguous vertical segments of a source color.
    Splits each segment into two halves (top/bottom).
    Keeps the top half as the source color and recolors the bottom half to a target color.
    The source and target colors are inferred from the first training pair.
    """
    def try_infer_params(inp, out):
        # We need to find source and target colors
        # and verify if splitting vertical segments works.
        unique_in = np.unique(inp)
        unique_out = np.unique(out)
        non_bg_in = unique_in[unique_in != 0]
        if len(non_bg_in) == 0: return None
        
        # Source color is likely the one that exists in both and is non-zero
        c_src = -1
        for c in non_bg_in:
            if c in unique_out:
                c_src = int(c); break
        if c_src == -1: return None
        
        # Target color is the one that exists in out but not in the same places in inp
        non_bg_out = [int(c) for c in unique_out if c != 0 and c != c_src]
        if not non_bg_out: return None
        c_tgt = non_bg_out[0]
        return c_src, c_tgt

    if not solver.pairs: return None
    params = try_infer_params(solver.train_in[0], solver.train_out[0])
    if params is None: return None
    c_src, c_tgt = params

    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        out = grid.copy()
        for c in range(cols):
            r = 0
            while r < rows:
                if grid[r, c] == c_src:
                    start_r = r
                    while r < rows and grid[r, c] == c_src: r += 1
                    end_r = r - 1
                    height = end_r - start_r + 1
                    num_stay = (height + 1) // 2
                    for r_idx in range(start_r + num_stay, end_r + 1):
                        out[r_idx, c] = c_tgt
                else: r += 1
        return out

    for inp, out_expected in solver.pairs:
        if not np.array_equal(apply_logic(inp), out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
