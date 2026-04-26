import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_project_objects_to_borders(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a central object (color 8). Finds other colored objects
    and projects them onto the borders of a new grid.
    The output grid size is the size of color 8's bounding box plus 2.
    Objects on the same side of color 8 are distributed to opposite borders.
    """
    def get_bbox(mask):
        rows, cols = np.where(mask)
        if len(rows) == 0: return None
        return rows.min(), rows.max(), cols.min(), cols.max()

    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # 1. Bbox of color 8
        bbox8 = get_bbox(grid == 8)
        if not bbox8: return None
        r1_8, r2_8, c1_8, c2_8 = bbox8
        S_r = r2_8 - r1_8 + 1
        S_c = c2_8 - c1_8 + 1
        
        out_h, out_w = S_r + 2, S_c + 2
        out = np.zeros((out_h, out_w), dtype=int)
        
        # Place 8
        for r in range(r1_8, r2_8 + 1):
            for c in range(c1_8, c2_8 + 1):
                if grid[r, c] == 8:
                    out[r - r1_8 + 1, c - c1_8 + 1] = 8
                    
        # 2. Find other colors
        colors = [c for c in np.unique(grid) if c != 0 and c != 8]
        
        # Group by side
        above, below, left, right = [], [], [], []
        for c in colors:
            xb = get_bbox(grid == c)
            xr1, xr2, xc1, xc2 = xb
            if xr2 < r1_8: above.append((abs(xr2 - r1_8), c, xb))
            elif xr1 > r2_8: below.append((abs(xr1 - r2_8), c, xb))
            elif xc2 < c1_8: left.append((abs(xc2 - c1_8), c, xb))
            elif xc1 > c2_8: right.append((abs(xc1 - c2_8), c, xb))
            
        # Mapping logic (Closer -> Opposite border, Farther -> Same border?)
        # Let's check T0: below has 4 (dist 3) and 3 (dist 8).
        # 4 (closer) -> Top. 3 (farther) -> Bottom.
        for dist, color, xb in sorted(below):
            if dist == min(d for d, _, _ in below): # Closer
                # Map to TOP border
                for c in range(xb[2], xb[3] + 1):
                    out[0, c - c1_8 + 1] = color
            else: # Farther
                # Map to BOTTOM border
                for c in range(xb[2], xb[3] + 1):
                    out[out_h-1, c - c1_8 + 1] = color
                    
        # T2: above has 7 (farther) and 1 (closer).
        # 7 (farther) -> Top. 1 (closer) -> Bottom.
        for dist, color, xb in sorted(above):
            if dist == max(d for d, _, _ in above): # Farther
                for c in range(xb[2], xb[3] + 1):
                    out[0, c - c1_8 + 1] = color
            else: # Closer
                for c in range(xb[2], xb[3] + 1):
                    out[out_h-1, c - c1_8 + 1] = color
                    
        # Left/Right similarly
        for dist, color, xb in sorted(right):
            if dist == min(d for d, _, _ in right): # Closer
                for r in range(xb[0], xb[1] + 1):
                    out[r - r1_8 + 1, 0] = color
            else: # Farther
                for r in range(xb[0], xb[1] + 1):
                    out[r - r1_8 + 1, out_w-1] = color
                    
        for dist, color, xb in sorted(left):
            if dist == min(d for d, _, _ in left): # Closer
                for r in range(xb[0], xb[1] + 1):
                    out[r - r1_8 + 1, out_w-1] = color
            else: # Farther
                for r in range(xb[0], xb[1] + 1):
                    out[r - r1_8 + 1, 0] = color
                    
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
