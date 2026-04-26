import numpy as np
from typing import List, Optional

def solve_pixel_upscaling_by_3x3(solver) -> Optional[List[np.ndarray]]:
    """
    Upscales each pixel (r, c) in the input grid into a 3x3 block of the same color
    in the output grid. The output size is (3*h) x (3*w).
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        out = np.zeros((3*h, 3*w), dtype=int)
        for r in range(h):
            for c in range(w):
                color = grid[r, c]
                if color != 0:
                    out[r*3 : (r+1)*3, c*3 : (c+1)*3] = color
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
