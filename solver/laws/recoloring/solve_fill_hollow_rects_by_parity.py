import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_fill_hollow_rects_by_parity(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies all connected components of color 1 that form hollow rectangular frames.
    Fills the interior of each frame with a color based on the parity of its size.
    If the interior height is even, it fills with color 2.
    If the interior height is odd, it fills with color 7.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        out = grid.copy()
        mask_1 = (grid == 1).astype(int)
        labeled, num_f = label(mask_1)
        
        for i in range(1, num_f + 1):
            comp_mask = (labeled == i)
            rows, cols = np.where(comp_mask)
            r1, r2 = rows.min(), rows.max()
            c1, c2 = cols.min(), cols.max()
            
            # Interior dimensions
            ih = (r2 - r1 + 1) - 2
            iw = (c2 - c1 + 1) - 2
            
            if ih > 0 and iw > 0:
                # Based on observations: Even height -> color 2, Odd height -> color 7
                target_color = 2 if (ih % 2 == 0) else 7
                # Fill the interior
                out[r1 + 1 : r2, c1 + 1 : c2] = target_color
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
