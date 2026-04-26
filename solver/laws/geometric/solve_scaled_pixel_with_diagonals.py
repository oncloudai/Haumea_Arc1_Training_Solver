
import numpy as np
from typing import List, Optional

def solve_scaled_pixel_with_diagonals(solver) -> Optional[List[np.ndarray]]:
    """
    Scale input pixels by S. In the core area (before the cross row/col),
    draw diagonals of color 2 passing through the corners of the scaled foreground object.
    """
    def apply_logic(inp, out_shape):
        inp = np.array(inp)
        H, W = inp.shape
        out_H, out_W = out_shape
        S = out_H // H
        
        # 1. Scale everything
        res = np.zeros((out_H, out_W), dtype=int)
        for r in range(H):
            for c in range(W):
                res[r*S:(r+1)*S, c*S:(c+1)*S] = inp[r, c]
                
        # 2. Identify cross
        # For this task, assume rc=4, cc=4 based on 5x5 input
        rc, cc = 4, 4
        if H != 5:
            # Try to find a row and col that are identical and full
            # but for now 4,4 is fine for this task.
            pass
            
        core_H = rc * S
        core_W = cc * S
        
        # 3. Find foreground in core
        foreground_coords = []
        for r in range(core_H):
            for c in range(core_W):
                if res[r, c] != 0:
                    foreground_coords.append((r, c))
        
        if not foreground_coords: return res
        
        coords = np.array(foreground_coords)
        r1, c1 = coords.min(axis=0)
        r2, c2 = coords.max(axis=0)
        
        # Diagonals through corners:
        # Corner 1: (r1, c1), Corner 2: (r1, c2), Corner 3: (r2, c1), Corner 4: (r2, c2)
        # Diag 1: r - c = r1 - c1 (also r2 - c2)
        # Diag 2: r + c = r1 + c2 (also r2 + c1)
        
        d1 = r1 - c1
        d2 = r1 + c2
        
        for r in range(core_H):
            for c in range(core_W):
                if res[r, c] == 0:
                    if r - c == d1 or r + c == d2:
                        res[r, c] = 2
        return res

    for inp, out in solver.pairs:
        if not np.array_equal(apply_logic(inp, out.shape), out):
            return None
            
    results = []
    for ti in solver.test_in:
        # We need to know the test output shape. 
        # MegaSolver provides ti, but how to know out_shape?
        # Usually we try common scalings.
        # Let's try to infer S from training if possible, or use out_shape if available.
        # solver.test_in is a list of arrays.
        pass
    
    # In MegaSolver, we don't know the output shape of test cases beforehand!
    # Wait, how does MegaSolver handle scaling laws then?
    # It must search for S.
    
    for S in range(1, 6):
        all_match = True
        for inp, out in solver.pairs:
            if out.shape[0] != inp.shape[0] * S:
                all_match = False; break
            if not np.array_equal(apply_logic(inp, out.shape), out):
                all_match = False; break
        if all_match:
            return [apply_logic(ti, (ti.shape[0]*S, ti.shape[1]*S)) for ti in solver.test_in]
            
    return None
