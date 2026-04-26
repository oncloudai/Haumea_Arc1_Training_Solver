import numpy as np
from typing import List, Optional

def solve_subgrid_by_square_frame(solver) -> Optional[List[np.ndarray]]:
    consistent = True; found_any = False
    for inp, out in solver.pairs:
        h, w = inp.shape
        found_color = None
        for color in range(1, 10):
            coords = np.argwhere(inp == color)
            if len(coords) == 0: continue
            
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            sh, sw = r_max - r_min + 1, c_max - c_min + 1
            
            if sh == sw and len(coords) == 4 * sh - 4:
                # Potential square frame. Check if it's actually a frame?
                # For now, let's just see if the subgrid matches the output.
                sub = inp[r_min:r_max+1, c_min:c_max+1]
                if sub.shape == out.shape and np.array_equal(sub, out):
                    found_color = color; break
        
        if found_color is None: consistent = False; break
        found_any = True
        
    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            h, w = ti.shape
            found_res = None
            for color in range(1, 10):
                coords = np.argwhere(ti == color)
                if len(coords) == 0: continue
                r_min, c_min = coords.min(axis=0)
                r_max, c_max = coords.max(axis=0)
                sh, sw = r_max - r_min + 1, c_max - c_min + 1
                if sh == sw and len(coords) == 4 * sh - 4:
                    found_res = ti[r_min:r_max+1, c_min:c_max+1]
                    break
            if found_res is not None: results.append(found_res)
            else: results.append(ti.copy())
        return results
    return None
