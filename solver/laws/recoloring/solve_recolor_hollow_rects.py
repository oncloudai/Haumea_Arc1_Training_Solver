import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_recolor_hollow_rects(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies all connected components of a source color that form hollow rectangular frames.
    Recolors these frames to a target color.
    Learns source and target colors from training.
    """
    def is_hollow_rect(comp_mask, grid):
        rows, cols = np.where(comp_mask)
        r1, r2 = rows.min(), rows.max()
        c1, c2 = cols.min(), cols.max()
        sh, sw = r2 - r1 + 1, c2 - c1 + 1
        if sh < 2 or sw < 2: return False
        
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                is_on_edge = (r == r1 or r == r2 or c == c1 or c == c2)
                if is_on_edge:
                    if comp_mask[r, c] == 0: return False
                else:
                    if grid[r, c] != 0: return False
        return True

    # Learn source and target color
    source_color = -1
    target_color = -1
    for inp, out in solver.pairs:
        inp = np.array(inp)
        out = np.array(out)
        diff_mask = (inp != out)
        if np.any(diff_mask):
            # Pick first different pixel
            dr, dc = np.where(diff_mask)
            source_color = inp[dr[0], dc[0]]
            target_color = out[dr[0], dc[0]]
            break
            
    if source_color == -1: return None

    def apply_logic(grid):
        grid = np.array(grid)
        out = grid.copy()
        mask = (grid == source_color).astype(int)
        labeled, num_f = label(mask)
        for i in range(1, num_f + 1):
            comp_mask = (labeled == i)
            if is_hollow_rect(comp_mask, grid):
                out[comp_mask] = target_color
        return out

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
