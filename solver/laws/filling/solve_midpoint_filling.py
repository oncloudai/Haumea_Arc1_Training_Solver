
import numpy as np
from typing import List, Optional

def solve_one(inp):
    h, w = inp.shape
    out = inp.copy()
    for r in range(h):
        row = inp[r]
        nonzero = np.argwhere(row != 0).flatten()
        if len(nonzero) == 2:
            c1, c2 = nonzero
            color1 = row[c1]
            color2 = row[c2]
            
            mid = (c1 + c2) // 2
            # Fill from c1 to mid-1 with color1
            for c in range(min(c1, mid), max(c1, mid) + 1):
                if c != mid: out[r, c] = color1
            # Fill from mid+1 to c2 with color2
            for c in range(min(mid+1, c2), max(mid+1, c2) + 1):
                if c != mid: out[r, c] = color2
            # Midpoint is 5
            out[r, mid] = 5
    return out

def solve_midpoint_filling(solver) -> Optional[List[np.ndarray]]:
    """
    Finds two non-zero pixels in a row and fills the gap with their colors and a midpoint marker 5.
    """
    for inp, out in solver.pairs:
        pred = solve_one(inp)
        if not np.array_equal(pred, out):
            return None
            
    return [solve_one(ti) for ti in solver.test_in]
