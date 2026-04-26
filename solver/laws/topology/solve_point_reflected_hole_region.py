import numpy as np
from typing import List, Optional

def solve_point_reflected_hole_region(solver) -> Optional[List[np.ndarray]]:
    """
    Finds the bounding box of the 'hole' (region of 0s) in the input.
    The output is a grid of the same size as the hole.
    Each pixel in the output is taken from the point-reflected position in the input grid.
    If the hole is at (r, c), the output pixel is grid[h-1-r, w-1-c].
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        zeros = np.where(grid == 0)
        if len(zeros[0]) == 0:
            return None
        
        r1, r2 = zeros[0].min(), zeros[0].max()
        c1, c2 = zeros[1].min(), zeros[1].max()
        
        out_h = r2 - r1 + 1
        out_w = c2 - c1 + 1
        out = np.zeros((out_h, out_w), dtype=int)
        
        for i in range(out_h):
            for j in range(out_w):
                # Corresponding input coordinates
                r = r1 + i
                c = c1 + j
                # Point reflect
                tr = h - 1 - r
                tc = w - 1 - c
                out[i, j] = grid[tr, tc]
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
