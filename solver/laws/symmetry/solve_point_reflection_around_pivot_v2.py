
import numpy as np
from typing import List, Optional

def solve_point_reflection_around_pivot_v2(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a 3x3 cross object as a pivot.
    Reflects all other non-zero pixels around the center of this pivot.
    """
    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        unique = np.unique(grid)
        colors = unique[unique != 0]
        
        pivot_center = None
        pivot_color = -1
        
        # Look for a 3x3 cross (5 pixels)
        for c in colors:
            coords = np.argwhere(grid == c)
            if len(coords) == 5:
                r_min, c_min = coords.min(axis=0)
                r_max, c_max = coords.max(axis=0)
                if r_max - r_min == 2 and c_max - c_min == 2:
                    # Check if it's a cross
                    mid_r, mid_c = (r_min + r_max) // 2, (c_min + c_max) // 2
                    is_cross = True
                    for r, cc in coords:
                        if not (r == mid_r or cc == mid_c):
                            is_cross = False; break
                    if is_cross:
                        pivot_center = (mid_r, mid_c)
                        pivot_color = c
                        break
        
        if pivot_center is None:
            return None
            
        output = grid.copy()
        pr, pc = pivot_center
        
        for r in range(rows):
            for c in range(cols):
                val = grid[r, c]
                if val != 0 and val != pivot_color:
                    # Reflect point (r, c) around (pr, pc)
                    # (r', c') = (2*pr - r, 2*pc - c)
                    nr, nc = 2*pr - r, 2*pc - c
                    if 0 <= nr < rows and 0 <= nc < cols:
                        output[nr, nc] = val
                        
        return output

    # Verify on pairs
    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    test_preds = []
    for inp in solver.test_in:
        pred = run_single(inp)
        test_preds.append(pred if pred is not None else np.array(inp))
        
    return test_preds
