
import numpy as np
from typing import List, Optional

def solve_one(inp):
    h, w = inp.shape
    # Try all possible subgrid sizes
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            if ph == h and pw == w: continue
            if h % ph == 0 and w % pw == 0:
                block = inp[:ph, :pw]
                
                # Check simple tiling
                tiled = np.tile(block, (h // ph, w // pw))
                if np.array_equal(tiled, inp):
                    return block
                
                # Check reflective tiling
                valid = True
                for ir in range(h // ph):
                    for ic in range(w // pw):
                        sub = inp[ir*ph:(ir+1)*ph, ic*pw:(ic+1)*pw]
                        target = block.copy()
                        if ir % 2 == 1: target = np.flipud(target)
                        if ic % 2 == 1: target = np.fliplr(target)
                        if not np.array_equal(sub, target):
                            valid = False; break
                    if not valid: break
                if valid:
                    return block
    return None

def solve_reflective_periodic_extraction(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies the smallest repeating block, allowing for simple or reflective tiling.
    """
    for inp, out in solver.pairs:
        pred = solve_one(inp)
        if pred is None or not np.array_equal(pred, out):
            return None
            
    return [solve_one(ti) for ti in solver.test_in]
