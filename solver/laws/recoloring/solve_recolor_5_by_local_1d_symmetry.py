import json
import numpy as np
from typing import List, Optional

def solve_recolor_5_by_local_1d_symmetry(solver) -> Optional[List[np.ndarray]]:
    """
    If a 5 pixel is part of an odd-length symmetric segment of 2s and 5s 
    centered at some point (which could be the 5 itself, or a 2).
    """
    def apply_logic(grid):
        grid = np.array(grid)
        H, W = grid.shape
        output = grid.copy()
        azure = set()
        
        # Check all possible odd-length horizontal segments
        for r in range(H):
            for L in range(3, W + 1, 2):
                for start in range(W - L + 1):
                    seg = grid[r, start : start + L]
                    if 2 in seg and all(p in [2, 5] for p in seg):
                        if all(seg[i] == seg[-(i+1)] for i in range(L//2)):
                            # All 5s in this symmetric segment turn 8
                            for c in range(start, start + L):
                                if grid[r, c] == 5: azure.add((r, c))
                                
        # Check all possible odd-length vertical segments
        for c in range(W):
            for L in range(3, H + 1, 2):
                for start in range(H - L + 1):
                    seg = grid[start : start + L, c]
                    if 2 in seg and all(p in [2, 5] for p in seg):
                        if all(seg[i] == seg[-(i+1)] for i in range(L//2)):
                            for r in range(start, start + L):
                                if grid[r, c] == 5: azure.add((r, c))
        
        for r, c in azure:
            output[r, c] = 8
        return output

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
