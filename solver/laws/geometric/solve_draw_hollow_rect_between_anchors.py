import numpy as np
from typing import List, Optional

def solve_draw_hollow_rect_between_anchors(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies anchor pixels (color 5) and draws a hollow rectangle of the main object's color
    inside the bounding box defined by these anchors.
    Specifically, the rectangle spans [r_min + 1, r_max - 1] and [c_min + 1, c_max - 1].
    """
    def apply_logic(grid):
        grid = np.array(grid)
        color_5_coords = np.where(grid == 5)
        if len(color_5_coords[0]) == 0: return None
        
        r_min, r_max = color_5_coords[0].min(), color_5_coords[0].max()
        c_min, c_max = color_5_coords[1].min(), color_5_coords[1].max()
        
        # Main color is the non-zero, non-5 color
        other_colors = set(np.unique(grid)) - {0, 5}
        if not other_colors: return None
        target_color = list(other_colors)[0]
        
        out = grid.copy()
        r1, r2 = r_min + 1, r_max - 1
        c1, c2 = c_min + 1, c_max - 1
        
        if r1 > r2 or c1 > c2: return None
        
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if r == r1 or r == r2 or c == c1 or c == c2:
                    out[r, c] = target_color
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
