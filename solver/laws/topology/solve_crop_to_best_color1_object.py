import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_crop_to_best_color1_object(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies all connected components of color 1.
    For each component, counts the number of color 2 pixels in its bounding box.
    Crops the grid to the bounding box of the component with the maximum count.
    In the resulting crop, only colors 1 and 2 are preserved; others are set to 0.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        mask_1 = (grid == 1).astype(int)
        # Using 8-connectivity for Color 1 objects
        labeled, num_f = label(mask_1, structure=np.ones((3,3)))
        if num_f == 0: return None
        
        best_bbox = None
        max_count = -1
        
        for i in range(1, num_f + 1):
            comp_mask = (labeled == i)
            rows, cols = np.where(comp_mask)
            r1, r2 = rows.min(), rows.max()
            c1, c2 = cols.min(), cols.max()
            
            # Count color 2 in this bbox
            sub = grid[r1:r2+1, c1:c2+1]
            count = np.sum(sub == 2)
            
            if count > max_count:
                max_count = count
                best_bbox = (r1, r2, c1, c2)
            elif count == max_count and best_bbox is not None:
                # Tie-break: could use size or position, but let's keep the first one
                pass
                
        if best_bbox is None: return None
        
        r1, r2, c1, c2 = best_bbox
        out = grid[r1:r2+1, c1:c2+1].copy()
        
        # Keep only 1 and 2
        mask_keep = (out == 1) | (out == 2)
        out[~mask_keep] = 0
        
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
