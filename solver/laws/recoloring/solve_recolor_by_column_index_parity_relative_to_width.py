import numpy as np
from typing import List, Optional

def solve_recolor_by_column_index_parity_relative_to_width(solver) -> Optional[List[np.ndarray]]:
    """
    Recolors pixels of a specific color (usually 5) based on the parity of their column index 
    relative to the total width of the grid.
    If (column_index + total_columns) is odd, color is changed to a target color (usually 3).
    Otherwise, it stays the original color.
    """
    def apply_logic(grid, c_src, c_tgt):
        grid = np.array(grid)
        rows, cols = grid.shape
        out = grid.copy()
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == c_src:
                    if (c + cols) % 2 == 1:
                        out[r, c] = c_tgt
        return out

    def try_infer_params(inp, out):
        unique_in = np.unique(inp)
        non_bg_in = unique_in[unique_in != 0]
        if len(non_bg_in) != 1: return None
        c_src = int(non_bg_in[0])
        
        unique_out = np.unique(out)
        non_bg_out = [int(c) for c in unique_out if c != 0 and c != c_src]
        if not non_bg_out: return None
        c_tgt = non_bg_out[0]
        
        # Verify the rule
        if np.array_equal(apply_logic(inp, c_src, c_tgt), out):
            return c_src, c_tgt
        return None

    if not solver.pairs: return None
    params = try_infer_params(solver.train_in[0], solver.train_out[0])
    if params is None: return None
    c_src, c_tgt = params

    for inp, out_expected in solver.pairs:
        if not np.array_equal(apply_logic(inp, c_src, c_tgt), out_expected):
            return None
            
    return [apply_logic(ti, c_src, c_tgt) for ti in solver.test_in]
