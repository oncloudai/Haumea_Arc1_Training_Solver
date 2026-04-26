import numpy as np
from typing import List, Optional

def solve_restore_interrupted_lines(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies non-background colors forming vertical or horizontal lines.
    Detects which lines are 'interrupted' by other colors or gaps.
    Restores the interrupted lines to be continuous in the output.
    If no lines are interrupted, defaults to making vertical lines continuous.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        
        unique_colors = set(np.unique(grid)) - {0}
        color_info = {}
        for c in unique_colors:
            pixels = np.argwhere(grid == c)
            r_min, c_min = pixels.min(axis=0)
            r_max, c_max = pixels.max(axis=0)
            h = r_max - r_min + 1
            w = c_max - c_min + 1
            
            # Heuristic: V if height >= width, H otherwise.
            role = 'V' if h >= w else 'H'
            color_info[c] = {
                'role': role,
                'bbox': (r_min, r_max, c_min, c_max)
            }
            
        output = grid.copy()
        interrupted_colors = set()
        
        for c, info in color_info.items():
            r_min, r_max, c_min, c_max = info['bbox']
            is_interrupted = False
            for r in range(r_min, r_max + 1):
                for c_idx in range(c_min, c_max + 1):
                    if grid[r, c_idx] != c:
                        is_interrupted = True; break
                if is_interrupted: break
            if is_interrupted: interrupted_colors.add(c)
                
        if interrupted_colors:
            for c in interrupted_colors:
                r_min, r_max, c_min, c_max = color_info[c]['bbox']
                output[r_min:r_max+1, c_min:c_max+1] = c
        else:
            # Heuristic for Ex 1: Pick V as winner
            for c, info in color_info.items():
                if info['role'] == 'V':
                    r_min, r_max, c_min, c_max = info['bbox']
                    output[r_min:r_max+1, c_min:c_max+1] = c
                    
        return output

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
