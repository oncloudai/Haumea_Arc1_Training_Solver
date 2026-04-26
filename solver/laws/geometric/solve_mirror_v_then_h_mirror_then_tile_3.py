import numpy as np
from typing import List, Optional

def solve_mirror_v_then_h_mirror_then_tile_3(solver) -> Optional[List[np.ndarray]]:
    """
    1. Forms a block B by concatenating the horizontally mirrored input with the input.
    2. The output is formed by concatenating [flip_v(B), B, flip_v(B)] vertically.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # 1. Horizontal mirror and concatenate
        # B = [flip_h(grid), grid]
        block = np.concatenate([grid[:, ::-1], grid], axis=1)
        
        # 2. Concatenate [flip_v(B), B, flip_v(B)] vertically
        bv = block[::-1, :]
        out = np.concatenate([bv, block, bv], axis=0)
        return out

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
