
import numpy as np
from typing import List, Optional

def solve_marker_pattern_recoloring(solver) -> Optional[List[np.ndarray]]:
    bg = 0
    # marker_color -> list of (dr, dc, stamp_color)
    marker_rules = {}
    consistent = True
    for inp, out in solver.pairs:
        h, w = inp.shape
        # Identify non-bg pixels in input
        markers = np.argwhere(inp != bg)
        if len(markers) == 0: continue
        
        for r, c in markers:
            color = inp[r, c]
            added = []
            # Look for pixels in 'out' that were 'bg' in 'inp' and are NOT 'bg' now
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if inp[nr, nc] == bg and out[nr, nc] != bg:
                            added.append((dr, dc, out[nr, nc]))
            
            if added:
                added_tuple = tuple(sorted(added))
                if color in marker_rules:
                    if marker_rules[color] != added_tuple:
                        consistent = False; break
                else:
                    marker_rules[color] = added_tuple
        if not consistent: break
        
    if consistent and marker_rules:
        def process(grid):
            res = grid.copy(); h, w = grid.shape
            for r in range(h):
                for c in range(w):
                    if grid[r, c] in marker_rules:
                        for dr, dc, stamp_color in marker_rules[grid[r, c]]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if res[nr, nc] == bg:
                                    res[nr, nc] = stamp_color
            return res
            
        # Final verification on all training pairs
        for inp, out in solver.pairs:
            if not np.array_equal(process(inp), out):
                consistent = False; break
        
        if consistent:
            return [process(ti) for ti in solver.test_in]
            
    return None
