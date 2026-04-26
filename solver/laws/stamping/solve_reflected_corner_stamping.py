import numpy as np
from typing import List, Optional

def solve_reflected_corner_stamping(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a small bounding box of colored pixels. Each pixel is reflected
    to a corner of the grid relative to the bounding box. The stamp size is 
    determined by the bounding box size, capped by the grid boundaries.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        rows, cols = np.where(grid != 0)
        if len(rows) == 0: return None
        
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()
        
        SR = r_max - r_min + 1
        SC = c_max - c_min + 1
        
        out = np.zeros_like(grid)
        
        for r, c in zip(rows, cols):
            color = grid[r, c]
            ri = r - r_min
            ci = c - c_min
            
            # Target Row
            if ri == 1:
                r_start = r_min - SR
                sh = SR
                if r_start < 0:
                    sh += r_start
                    r_start = 0
            else:
                r_start = r_max + 1
                sh = SR
                if r_start + sh > h:
                    sh = h - r_start
            
            # Target Col
            if ci == 1:
                c_start = c_min - SC
                sw = SC
                if c_start < 0:
                    sw += c_start
                    c_start = 0
            else:
                c_start = c_max + 1
                sw = SC
                if c_start + sw > w:
                    sw = w - c_start
            
            if sh > 0 and sw > 0:
                out[r_start : r_start + sh, c_start : c_start + sw] = color
                
        # Keep original pixels
        for r, c in zip(rows, cols):
            out[r, c] = grid[r, c]
            
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
