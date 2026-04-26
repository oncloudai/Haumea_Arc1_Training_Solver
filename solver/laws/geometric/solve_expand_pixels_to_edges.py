
import numpy as np
from typing import List, Optional

def solve_expand_pixels_to_edges(solver) -> Optional[List[np.ndarray]]:
    """
    Expands each pixel (r, c) of the input grid into a (H+2, W+2) output grid.
    Each pixel maps to (r+1, c+1) in the output.
    Pixels on the edges of the input grid also expand to the outer boundary
    of the output grid.
    """
    def apply_logic(inp):
        inp = np.array(inp)
        h, w = inp.shape
        out_h, out_w = h + 2, w + 2
        out = np.zeros((out_h, out_w), dtype=int)
        
        for r in range(h):
            for c in range(w):
                v = inp[r, c]
                out[r+1, c+1] = v
                if r == 0: out[r, c+1] = v
                if r == h - 1: out[r+2, c+1] = v
                if c == 0: out[r+1, c] = v
                if c == w - 1: out[r+1, c+2] = v
        return out

    for inp, out in solver.pairs:
        if not np.array_equal(apply_logic(inp), out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
