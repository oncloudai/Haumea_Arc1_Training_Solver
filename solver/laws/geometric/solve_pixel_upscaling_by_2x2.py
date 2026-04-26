import numpy as np
from typing import List, Optional

def solve_pixel_upscaling_by_2x2(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies the bounding box of all non-zero pixels in the input.
    The output is a 2x upscaled version of this bounding box, 
    where each pixel becomes a 2x2 block of the same color.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = np.where(grid != 0)
        if len(rows) == 0: return None
        
        r1, r2 = rows.min(), rows.max()
        c1, c2 = cols.min(), cols.max()
        
        sub = grid[r1:r2+1, c1:c2+1]
        oh, ow = sub.shape
        
        out = np.zeros((oh * 2, ow * 2), dtype=int)
        for r in range(oh):
            for c in range(ow):
                if sub[r, c] != 0:
                    out[2*r:2*r+2, 2*c:2*c+2] = sub[r, c]
                    
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
