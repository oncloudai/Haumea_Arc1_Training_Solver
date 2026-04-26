import numpy as np
from typing import List, Optional
from scipy.ndimage import binary_fill_holes

def solve_fill_enclosed_zeros_with_1(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies all 0s in the input grid.
    Fills those 0s with color 1 if they are 'enclosed' (not reachable from the edge).
    This is equivalent to filling holes in the binary mask of non-zero pixels.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        # Create mask where non-zero pixels are 1
        mask = (grid != 0).astype(int)
        
        # Fill holes in the mask
        filled_mask = binary_fill_holes(mask).astype(int)
        
        # The new 1s are where filled_mask is 1 but grid is 0
        out = grid.copy()
        out[(grid == 0) & (filled_mask == 1)] = 1
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
