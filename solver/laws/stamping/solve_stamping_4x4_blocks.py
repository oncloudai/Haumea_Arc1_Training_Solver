
import numpy as np
from typing import List, Optional

def solve_stamping_4x4_blocks(solver) -> Optional[List[np.ndarray]]:
    """
    For each non-zero pixel in the input, place a 4x4 block of the same color
    in the output grid. The output grid is twice the size of the input.
    The top-left corner of the block is at (2*r - 2, 2*c - 2).
    """
    def apply_logic(inp):
        inp = np.array(inp)
        H, W = inp.shape
        out_H, out_W = H * 2, W * 2
        out = np.zeros((out_H, out_W), dtype=int)
        
        coords = np.argwhere(inp != 0)
        for r, c in coords:
            color = inp[r, c]
            r_out = 2 * r - 2
            c_out = 2 * c - 2
            for dr in range(4):
                for dc in range(4):
                    nr, nc = r_out + dr, c_out + dc
                    if 0 <= nr < out_H and 0 <= nc < out_W:
                        out[nr, nc] = color
        return out

    for inp, out in solver.pairs:
        if not np.array_equal(apply_logic(inp), out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
