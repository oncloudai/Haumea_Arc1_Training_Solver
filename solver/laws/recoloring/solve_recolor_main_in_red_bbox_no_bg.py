import numpy as np
from typing import List, Optional

def solve_recolor_main_in_red_bbox_no_bg(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies rectangular regions (at least 2x2) that contain only Red (2) and one
    main color, such that Red pixels define the extreme boundaries of the rectangle
    and no background (0) is present inside. Main color pixels in these regions
    turn Yellow (4).
    """
    def apply(grid):
        out = grid.copy()
        unq = np.unique(grid)
        # Red is always 2. Identify the other color(s).
        others = [c for c in unq if c != 0 and c != 2]
        if not others: return out
        # Usually there's one main color besides red and background.
        main_color = others[0]
        
        rows, cols = grid.shape
        to_change = np.zeros((rows, cols), dtype=bool)
        
        # Iterate over all possible rectangles
        for r1 in range(rows):
            for r2 in range(r1 + 1, rows): # At least 2 rows
                for c1 in range(cols):
                    for c2 in range(c1 + 1, cols): # At least 2 cols
                        sub = grid[r1:r2+1, c1:c2+1]
                        
                        # 1. No background color (0) allowed
                        if 0 in sub: continue
                        
                        # 2. Find Red pixels (2)
                        red_coords = np.argwhere(sub == 2)
                        if red_coords.size == 0: continue
                        
                        # 3. Check tight Red BBox
                        rmin, cmin = red_coords.min(axis=0)
                        rmax, cmax = red_coords.max(axis=0)
                        
                        if rmin == 0 and rmax == (r2 - r1) and cmin == 0 and cmax == (c2 - c1):
                            to_change[r1:r2+1, c1:c2+1] = True
                            
        out[to_change & (grid == main_color)] = 4
        return out

    for inp, out_expected in solver.pairs:
        pred = apply(inp)
        if not np.array_equal(pred, out_expected):
            return None
            
    results = []
    for ti in solver.test_in:
        results.append(apply(ti))
    return results
