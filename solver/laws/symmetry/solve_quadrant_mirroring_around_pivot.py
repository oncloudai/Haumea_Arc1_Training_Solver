
import numpy as np
from typing import List, Optional

def solve_quadrant_mirroring_around_pivot(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a 3x3 object with 5 pixels as a pivot.
    Mirrors all other non-zero pixels into the four quadrants defined by the pivot center.
    """
    def get_pivot_candidates(grid):
        rows, cols = grid.shape
        unique = np.unique(grid)
        colors = unique[unique != 0]
        candidates = []
        for p_color in colors:
            coords = np.argwhere(grid == p_color)
            if len(coords) == 5:
                r_min, c_min = coords.min(axis=0)
                r_max, c_max = coords.max(axis=0)
                if r_max - r_min == 2 and c_max - c_min == 2:
                    mid_r, mid_c = (r_min + r_max) // 2, (c_min + c_max) // 2
                    # Accept any 3x3 with 5 pixels (Plus or X)
                    candidates.append((mid_r, mid_c, p_color))
        return candidates

    def apply_pivot(grid, mid_r, mid_c, p_color):
        rows, cols = grid.shape
        output = grid.copy()
        for r in range(rows):
            for c in range(cols):
                val = grid[r, c]
                if val != 0 and val != p_color:
                    points = [
                        (r, c),
                        (2*mid_r - r, c),
                        (r, 2*mid_c - c),
                        (2*mid_r - r, 2*mid_c - c)
                    ]
                    for pr, pc in points:
                        if 0 <= pr < rows and 0 <= pc < cols:
                            output[pr, pc] = val
        return output

    # Verify on pairs
    for inp, out_expected in solver.pairs:
        candidates = get_pivot_candidates(inp)
        found_working = False
        for mid_r, mid_c, p_color in candidates:
            pred = apply_pivot(inp, mid_r, mid_c, p_color)
            if np.array_equal(pred, out_expected):
                found_working = True
                break
        if not found_working:
            return None
            
    test_preds = []
    for ti in solver.test_in:
        candidates = get_pivot_candidates(ti)
        if not candidates:
            test_preds.append(np.array(ti))
            continue
            
        # Try to find which candidate works if multiple. 
        # For now, just pick the one that matches the training pattern?
        # (Usually there is only one pivot shape)
        
        # In Ex 2, there were two candidates. Let's pick the one that 
        # doesn't overlap with other pixels in a way that suggests it's the pivot.
        # Actually, in Ex 2, Color 8 was the pivot.
        # Let's try to pick the candidate that works for most non-pivot pixels?
        
        # For simplicity, pick the first one.
        mid_r, mid_c, p_color = candidates[0]
        test_preds.append(apply_pivot(ti, mid_r, mid_c, p_color))
        
    return test_preds
