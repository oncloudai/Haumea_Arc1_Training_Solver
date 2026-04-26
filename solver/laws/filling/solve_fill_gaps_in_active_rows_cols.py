
import numpy as np
from typing import List, Optional

def solve_fill_gaps_in_active_rows_cols(solver) -> Optional[List[np.ndarray]]:
    """
    Find all rows and columns with >= threshold blue pixels (color 1).
    For each active row, fill all zeros within the global bounding box of color 1 with color 2.
    For each active column, fill all zeros within the global bounding box of color 1 with color 2.
    Threshold is typically 4.
    """
    # We'll try thresholds 3 and 4
    for threshold in [4, 3]:
        consistent = True
        found_any_fill = False
        for inp, out in solver.pairs:
            res = inp.copy()
            h, w = inp.shape
            
            blue_coords = np.argwhere(inp == 1)
            if len(blue_coords) == 0:
                if not np.array_equal(res, out): consistent = False; break
                continue
                
            r_min, c_min = blue_coords.min(axis=0)
            r_max, c_max = blue_coords.max(axis=0)
            
            active_rows = []
            for r in range(h):
                if np.sum(inp[r, :] == 1) >= threshold: active_rows.append(r)
            
            active_cols = []
            for c in range(w):
                if np.sum(inp[:, c] == 1) >= threshold: active_cols.append(c)
                
            local_found = False
            for r in active_rows:
                for c in range(c_min, c_max + 1):
                    if res[r, c] == 0:
                        res[r, c] = 2
                        local_found = True; found_any_fill = True
            
            for c in active_cols:
                for r in range(r_min, r_max + 1):
                    if res[r, c] == 0:
                        res[r, c] = 2
                        local_found = True; found_any_fill = True
            
            if not np.array_equal(res, out):
                consistent = False; break
        
        if consistent and found_any_fill:
            # Apply to test inputs
            results = []
            for ti in solver.test_in:
                res = ti.copy()
                h, w = ti.shape
                blue_coords = np.argwhere(ti == 1)
                if len(blue_coords) == 0:
                    results.append(res); continue
                r_min, c_min = blue_coords.min(axis=0)
                r_max, c_max = blue_coords.max(axis=0)
                
                for r in range(h):
                    if np.sum(ti[r, :] == 1) >= threshold:
                        for c in range(c_min, c_max + 1):
                            if res[r, c] == 0: res[r, c] = 2
                for c in range(w):
                    if np.sum(ti[:, c] == 1) >= threshold:
                        for r in range(r_min, r_max + 1):
                            if res[r, c] == 0: res[r, c] = 2
                results.append(res)
            return results
            
    return None
