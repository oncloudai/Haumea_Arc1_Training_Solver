import numpy as np
from typing import List, Optional

def solve_rotate_regions_90_180(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies three 3x3 regions separated by color 5 dividers.
    Region 1: C0-2, Region 2: C4-6, Region 3: C8-10.
    For each color X:
    If it's in Region 1, it's also in Reg 2 (rotated 90 CW) and Reg 3 (rotated 180).
    If it's in Region 2, it's also in Reg 3 (rotated 90 CW).
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        if w != 11 or h != 3: return None
        
        # Region boundaries
        r1 = grid[:, 0:3]
        r2 = grid[:, 4:7]
        r3 = grid[:, 8:11]
        
        out = grid.copy()
        
        # Colors in each region
        def get_colors(sub):
            return [c for c in np.unique(sub) if c != 0 and c != 5]
            
        c1 = get_colors(r1)
        c2 = get_colors(r2)
        
        # Propagate c1
        for color in c1:
            # Extract pattern
            p = (r1 == color).astype(int)
            # Reg 2: 90 CW
            p90 = np.rot90(p, -1)
            out[:, 4:7] = np.where(p90, color, out[:, 4:7])
            # Reg 3: 180
            p180 = np.rot90(p, -2)
            out[:, 8:11] = np.where(p180, color, out[:, 8:11])
            
        # Propagate c2
        for color in c2:
            p = (r2 == color).astype(int)
            # Reg 3: 90 CW
            p90 = np.rot90(p, -1)
            out[:, 8:11] = np.where(p90, color, out[:, 8:11])
            
        return out

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
