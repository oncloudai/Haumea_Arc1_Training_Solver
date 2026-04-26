import numpy as np
from typing import List, Optional
from scipy.ndimage import label, binary_fill_holes

def solve_border_and_hole_fill(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies objects of color 6, adds a 1-pixel border of color 3,
    and fills holes with color 4.
    Used for task 543a7ed5.
    """
    def run_single(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        out = grid.copy()
        
        # 1. Identify objects of color 6
        mask6 = (grid == 6)
        labeled, num_features = label(mask6)
        
        for i in range(1, num_features + 1):
            obj_mask = (labeled == i)
            coords = np.argwhere(obj_mask)
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            
            # 2. Add border of color 3
            br_min, bc_min = max(0, r_min - 1), max(0, c_min - 1)
            br_max, bc_max = min(rows - 1, r_max + 1), min(cols - 1, c_max + 1)
            
            for r in range(br_min, br_max + 1):
                for c in range(bc_min, bc_max + 1):
                    if out[r, c] == 8: # Background in this task is 8
                        out[r, c] = 3
            
            # 3. Fill holes of 6s with 4
            obj_h, obj_w = r_max - r_min + 1, c_max - c_min + 1
            obj_with_holes = np.zeros((obj_h, obj_w), dtype=bool)
            for r, c in coords:
                obj_with_holes[r - r_min, c - c_min] = True
            
            filled_obj = binary_fill_holes(obj_with_holes)
            holes = filled_obj & ~obj_with_holes
            
            for r_rel, c_rel in np.argwhere(holes):
                out[r_min + r_rel, c_min + c_rel] = 4
                
        return out

    results = []
    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    for ti in solver.test_in:
        results.append(run_single(ti))
    return results
