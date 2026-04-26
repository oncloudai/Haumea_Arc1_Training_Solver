import numpy as np
from typing import List, Optional

def solve_marker_midpoint_reflection(solver) -> Optional[List[np.ndarray]]:
    # 1. Infer Backgrounds and Marker from the first pair
    if not solver.pairs: return None
    inp0, out0 = solver.pairs[0]
    
    vals, counts = np.unique(inp0, return_counts=True)
    bg_in = vals[np.argmax(counts)]
    
    vals_out, counts_out = np.unique(out0, return_counts=True)
    bg_out = vals_out[np.argmax(counts_out)]
    
    unique_in = np.unique(inp0)
    unique_out = np.unique(out0)
    
    # Marker is in In but not Out (and not bg)
    candidates = [c for c in unique_in if c not in unique_out and c != bg_in]
    marker_color = candidates[0] if candidates else -1
    if marker_color == -1: return None

    def process(grid, m_color, b_in, b_out):
        # Find Marker
        m_coords = np.argwhere(grid == m_color)
        if len(m_coords) == 0: return None
        c_marker = np.mean(m_coords, axis=0)
        
        # Find Object (pixels not bg and not marker)
        obj_mask = (grid != b_in) & (grid != m_color)
        if not np.any(obj_mask): return None
        obj_coords = np.argwhere(obj_mask)
        c_obj = np.mean(obj_coords, axis=0)
        
        # Determine orientation
        dr = c_marker[0] - c_obj[0]
        dc = c_marker[1] - c_obj[1]
        
        res = np.full_like(grid, b_out)
        
        # Copy original object
        for r, c in obj_coords:
            res[r, c] = grid[r, c]
            
        # Determine Reflection Axis and Apply
        if abs(dr) > abs(dc):
            # Vertical Reflection (Horizontal Axis)
            sign = np.sign(dr) 
            axis_r = c_marker[0] - 0.5 * sign
            for r, c in obj_coords:
                nr = int(round(2 * axis_r - r))
                if 0 <= nr < res.shape[0]:
                    res[nr, c] = grid[r, c]
        else:
            # Horizontal Reflection (Vertical Axis)
            sign = np.sign(dc)
            axis_c = c_marker[1] - 0.5 * sign
            for r, c in obj_coords:
                nc = int(round(2 * axis_c - c))
                if 0 <= nc < res.shape[1]:
                    res[r, nc] = grid[r, c]
        return res

    for inp, out in solver.pairs:
        pred = process(inp, marker_color, bg_in, bg_out)
        if pred is None or not np.array_equal(pred, out): return None
        
    results = []
    for ti in solver.test_in:
        res = process(ti, marker_color, bg_in, bg_out)
        if res is None: return None
        results.append(res)
    return results
