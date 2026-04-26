import numpy as np
from typing import List, Optional

def solve_4x4_xor_to_color_3(solver) -> Optional[List[np.ndarray]]:
    """
    For a 9x4 input, splits it into two 4x4 halves (top and bottom, separated by a row).
    The output is a 4x4 grid of color 3 at positions where one half has a non-zero
    pixel and the other half is empty (bitwise XOR of presence).
    """
    def apply_logic(grid):
        grid = np.array(grid)
        if grid.shape != (9, 4): return None
        
        top = grid[:4, :]
        bottom = grid[5:, :]
        
        out = np.zeros((4, 4), dtype=int)
        for r in range(4):
            for c in range(4):
                v1 = top[r, c]
                v2 = bottom[r, c]
                if (v1 != 0 and v2 == 0) or (v1 == 0 and v2 != 0):
                    out[r, c] = 3
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
