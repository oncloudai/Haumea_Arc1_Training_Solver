import numpy as np
from typing import List, Optional

def solve_4x4_empty_overlap_to_color_2(solver) -> Optional[List[np.ndarray]]:
    """
    For an 8x4 input, splits it into two 4x4 halves (top and bottom).
    The output is a 4x4 grid where a pixel is color 2 if BOTH halves 
    have an empty (0) pixel at that position.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        if grid.shape != (8, 4): return None
        
        top = grid[:4, :]
        bottom = grid[4:, :]
        
        out = np.zeros((4, 4), dtype=int)
        for r in range(4):
            for c in range(4):
                if top[r, c] == 0 and bottom[r, c] == 0:
                    out[r, c] = 2
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
