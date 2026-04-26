import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_clear_lines_with_holes_by_orientation(solver) -> Optional[List[np.ndarray]]:
    """
    For each connected component of a single color:
    1. Find all 'holes' (zeros in the original grid) within its bounding box.
    2. If the component is wider than it is tall, clear all columns that contain a hole.
    3. If the component is taller than it is wide, clear all rows that contain a hole.
    Clearing means setting that row/column range within the component to 0.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        out = grid.copy()
        
        for color in range(1, 10):
            mask = (grid == color).astype(int)
            labeled, num_f = label(mask)
            for i in range(1, num_f + 1):
                comp_mask = (labeled == i)
                rows, cols = np.where(comp_mask)
                r1, r2 = rows.min(), rows.max()
                c1, c2 = cols.min(), cols.max()
                
                ch = r2 - r1 + 1
                cw = c2 - c1 + 1
                
                # Check for 0s in the component's bounding box
                sub = grid[r1 : r2 + 1, c1 : c2 + 1]
                rows_with_zero, cols_with_zero = np.where(sub == 0)
                
                if cw > ch:
                    # Wider: clear columns
                    for c_rel in np.unique(cols_with_zero):
                        out[r1 : r2 + 1, c1 + c_rel] = 0
                elif ch > cw:
                    # Taller: clear rows
                    for r_rel in np.unique(rows_with_zero):
                        out[r1 + r_rel, c1 : c2 + 1] = 0
                else:
                    # Square: clear both
                    for r_rel in np.unique(rows_with_zero):
                        out[r1 + r_rel, c1 : c2 + 1] = 0
                    for c_rel in np.unique(cols_with_zero):
                        out[r1 : r2 + 1, c1 + c_rel] = 0
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
