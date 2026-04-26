import numpy as np
from typing import List, Optional

def solve_concentric_boxes_repeating_dots(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies colors that define concentric boxes and repeating dot patterns on their edges.
    1. A single pixel of color X at (r, c) defines a set of 4 points:
       (r, c), (r, W-1-c), (H-1-r, c), (H-1-r, W-1-c).
    2. Two pixels of color X at (r1, c1) and (r2, c2) define a repeating pattern on edges.
       Usually, one coordinate matches an outer box edge and the other matches an inner box edge.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        out = np.zeros_like(grid)
        
        unique_colors = np.unique(grid)
        colors = [c for c in unique_colors if c != 0]
        
        for color in colors:
            coords = np.where(grid == color)
            pts = list(zip(coords[0], coords[1]))
            
            if len(pts) == 1:
                r, c = pts[0]
                syms = [(r, c), (r, w-1-c), (h-1-r, c), (h-1-r, w-1-c)]
                for sr, sc in syms:
                    out[sr, sc] = color
            elif len(pts) == 2:
                r1, c1 = pts[0]
                r2, c2 = pts[1]
                
                # These two points plus their reflections define the edges
                all_r = {r1, r2, h-1-r1, h-1-r2}
                all_c = {c1, c2, w-1-c1, w-1-c2}
                
                min_r, max_r = min(all_r), max(all_r)
                min_c, max_c = min(all_c), max(all_c)
                
                # Period
                dr = abs(r1 - r2)
                dc = abs(c1 - c2)
                p = max(dr, dc)
                if p == 0: continue
                
                # Fill edges
                for r in all_r:
                    for c in range(min_c + p, max_c, p):
                        out[r, c] = color
                for c in all_c:
                    for r in range(min_r + p, max_r, p):
                        out[r, c] = color
                
                # Also include the original points and their reflections
                syms1 = [(r1, c1), (r1, w-1-c1), (h-1-r1, c1), (h-1-r1, w-1-c1)]
                syms2 = [(r2, c2), (r2, w-1-c2), (h-1-r2, c2), (h-1-r2, w-1-c2)]
                for sr, sc in syms1 + syms2:
                    out[sr, sc] = color
                    
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
