import numpy as np
from typing import List, Optional

def solve_vertical_oscillating_mirror_tiling(solver) -> Optional[List[np.ndarray]]:
    """
    Implements vertical mirror-tiling with an oscillating row selection pattern.
    Row selection: 0, 1, 2, ..., H-1, H-2, ..., 0, 1, ...
    Target height is inferred from training examples.
    """
    def apply_logic(grid, target_h):
        grid = np.array(grid)
        h, w = grid.shape
        output_rows = []
        idx = 0
        direction = 1 # 1 for forward, -1 for backward
        
        while len(output_rows) < target_h:
            output_rows.append(grid[idx].copy())
            if h == 1: break # Avoid infinite loop
            if idx == h - 1: direction = -1
            elif idx == 0 and len(output_rows) > 1: direction = 1
            idx += direction
        return np.array(output_rows)

    def try_infer_height(inp, out):
        h_in, w_in = inp.shape
        h_out, w_out = out.shape
        # Common pattern: (h_in - 1) * repetitions + 1
        # Example: h_in=3, repetitions=4 => (2*4)+1 = 9.
        # Example: h_in=4, repetitions=4 => (3*4)+1 = 13.
        # Example: h_in=5, repetitions=4 => (4*4)+1 = 17.
        if h_in > 1 and (h_out - 1) % (h_in - 1) == 0:
            return h_out
        if h_out % h_in == 0:
            return h_out
        return None

    def infer_height_params(solver):
        # We need to see if there is a consistent rule for repetitions
        # Rule: h_out = (h_in - 1) * R + 1
        rs = []
        for inp, out in solver.pairs:
            h_in, h_out = inp.shape[0], out.shape[0]
            if h_in > 1 and (h_out - 1) % (h_in - 1) == 0:
                rs.append((h_out - 1) // (h_in - 1))
            else: return None
        if len(set(rs)) == 1: return rs[0]
        return None

    if not solver.pairs: return None
    R = infer_height_params(solver)
    if R is None: return None
    
    for inp, out_expected in solver.pairs:
        target_h = (inp.shape[0] - 1) * R + 1
        pred = apply_logic(inp, target_h)
        if not np.array_equal(pred, out_expected):
            return None
            
    results = []
    for ti in solver.test_in:
        target_h = (ti.shape[0] - 1) * R + 1
        results.append(apply_logic(ti, target_h))
    return results
