import numpy as np
from typing import List, Optional

def solve_marker_offset_composition(solver) -> Optional[List[np.ndarray]]:
    consistent = True; found_any = False
    for inp, out in solver.pairs:
        # Determine background color
        vals, counts = np.unique(inp, return_counts=True)
        bg = vals[np.argmax(counts)]
        
        other_colors = [v for v in vals if v != bg]
        if not other_colors: consistent = False; break
        
        # Collect all offsets
        all_offsets = {} # (dr, dc) -> color
        max_dr, max_dc = 0, 0
        
        for color in other_colors:
            coords = np.argwhere(inp == color)
            if len(coords) == 0: continue
            
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            # Use bbox center as reference
            cr, cc = (r_min + r_max) / 2, (c_min + c_max) / 2
            
            for r, c in coords:
                dr, dc = int(r - cr), int(c - cc)
                all_offsets[(dr, dc)] = color
                max_dr = max(max_dr, abs(dr))
                max_dc = max(max_dc, abs(dc))
        
        # Build prediction
        h_out, w_out = 2 * max_dr + 1, 2 * max_dc + 1
        if out.shape != (h_out, w_out):
            consistent = False; break
            
        pred = np.full((h_out, w_out), bg, dtype=int)
        for (dr, dc), color in all_offsets.items():
            pred[max_dr + dr, max_dc + dc] = color
            
        if not np.array_equal(pred, out):
            consistent = False; break
        found_any = True
        
    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            vals, counts = np.unique(ti, return_counts=True)
            bg = vals[np.argmax(counts)]
            other_colors = [v for v in vals if v != bg]
            
            all_offsets = {}
            max_dr, max_dc = 0, 0
            for color in other_colors:
                coords = np.argwhere(ti == color)
                if len(coords) == 0: continue
                r_min, c_min = coords.min(axis=0)
                r_max, c_max = coords.max(axis=0)
                cr, cc = (r_min + r_max) / 2, (c_min + c_max) / 2
                for r, c in coords:
                    dr, dc = int(r - cr), int(c - cc)
                    all_offsets[(dr, dc)] = color
                    max_dr = max(max_dr, abs(dr))
                    max_dc = max(max_dc, abs(dc))
            
            res = np.full((2 * max_dr + 1, 2 * max_dc + 1), bg, dtype=int)
            for (dr, dc), color in all_offsets.items():
                res[max_dr + dr, max_dc + dc] = color
            results.append(res)
        return results
    return None
