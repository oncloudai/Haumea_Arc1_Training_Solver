import numpy as np
from typing import List, Optional

def solve_reflected_dotted_lines_by_count(solver) -> Optional[List[np.ndarray]]:
    """
    For each color X:
    1. Find its input pixels. Generate 4-way reflections for each.
    2. If exactly TWO input pixels:
       For each pixel (r, c), connect it to its horizontal OR vertical mirror
       based on which coordinate is closer to the edge.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        out = np.zeros_like(grid)
        
        unique_colors = np.unique(grid)
        for color in unique_colors:
            if color == 0: continue
            coords = np.where(grid == color)
            pts = list(zip(coords[0], coords[1]))
            
            # Ensure reflections are always there
            all_pts = set()
            for r, c in pts:
                all_pts.add((r, c))
                all_pts.add((r, w-1-c))
                all_pts.add((h-1-r, c))
                all_pts.add((h-1-r, w-1-c))
            
            for r, c in all_pts:
                out[r, c] = color
                
            if len(pts) == 2:
                for r, c in pts:
                    # Edge distances
                    d_top = r
                    d_bot = h - 1 - r
                    d_left = c
                    d_right = w - 1 - c
                    min_dist = min(d_top, d_bot, d_left, d_right)
                    
                    if min_dist == d_top or min_dist == d_bot:
                        # Row is closer to edge. Connect horizontally.
                        c1, c2 = min(c, w - 1 - c), max(c, w - 1 - c)
                        for nc in range(c1, c2 + 1, 2):
                            out[r, nc] = color
                            out[h - 1 - r, nc] = color
                    else:
                        # Col is closer to edge. Connect vertically.
                        r1, r2 = min(r, h - 1 - r), max(r, h - 1 - r)
                        for nr in range(r1, r2 + 1, 2):
                            out[nr, c] = color
                            out[nr, w - 1 - c] = color
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
