import numpy as np
from typing import List, Optional

def solve_color_line_expansion(solver) -> Optional[List[np.ndarray]]:
    bg = 0; color_dirs = {} # color -> set of 'H', 'V'
    consistent = True
    for inp, out in solver.pairs:
        h, w = inp.shape
        # Identify directions for each color in THIS example
        for c in range(1, 10):
            in_coords = np.argwhere(inp == c)
            if len(in_coords) == 0: continue
            
            # Check if this color expanded horizontally
            has_H = False; has_V = False
            for r, ci in in_coords:
                if np.all(out[r, :] == c): has_H = True
                if np.all(out[:, ci] == c): has_V = True
            
            # Record directions
            dirs = []
            if has_H: dirs.append('H')
            if has_V: dirs.append('V')
            
            if dirs:
                t_dirs = tuple(sorted(dirs))
                if c in color_dirs and color_dirs[c] != t_dirs: consistent = False; break
                color_dirs[c] = t_dirs
        if not consistent: break
    
    if consistent and color_dirs:
        def process(grid):
            res = grid.copy(); h, w = grid.shape
            # For 178fcbfb, draw V first, then H
            for c, dirs in color_dirs.items():
                if 'V' in dirs:
                    for r, cc in np.argwhere(grid == c): res[:, cc] = c
            for c, dirs in color_dirs.items():
                if 'H' in dirs:
                    for rr, cc in np.argwhere(grid == c): res[rr, :] = c
            return res
        # Final check on all train examples
        for inp, out in solver.pairs:
            if not np.array_equal(process(inp), out): consistent = False; break
        if consistent:
            return [process(ti) for ti in solver.test_in]
    return None
