import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_fill_target_object_holes_with_2(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies all connected components of color 1.
    The 'target' object is the one that has no color 2 pixels in its bounding box.
    The output is the target object's grid, but with all its 0s replaced by color 2.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        mask_1 = (grid == 1).astype(int)
        labeled, num_f = label(mask_1)
        if num_f == 0: return None
        
        target_sub = None
        for i in range(1, num_f + 1):
            comp_mask = (labeled == i)
            rows, cols = np.where(comp_mask)
            r1, r2 = rows.min(), rows.max()
            c1, c2 = cols.min(), cols.max()
            
            # Check for color 2 in this bbox
            sub_2 = grid[r1:r2+1, c1:c2+1]
            if not np.any(sub_2 == 2):
                # This is our target!
                # Extract the pattern of 1s
                target_sub = np.zeros((r2-r1+1, c2-c1+1), dtype=int)
                for r in range(r1, r2+1):
                    for c in range(c1, c2+1):
                        if grid[r, c] == 1 and comp_mask[r, c]:
                            target_sub[r-r1, c-c1] = 1
                        else:
                            target_sub[r-r1, c-c1] = 2
                break
                
        return target_sub

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
