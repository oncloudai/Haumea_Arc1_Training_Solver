
import numpy as np
from typing import List, Optional

def solve_rotational_tiling_2x2(solver) -> Optional[List[np.ndarray]]:
    """
    Creates a 2x2 tiling of the input grid using its rotations to form
     a 4-fold rotationally symmetric output.
    TL = inp, TR = rot90_cw(inp), BR = rot180_cw(inp), BL = rot270_cw(inp)
    """
    def apply_logic(inp):
        inp = np.array(inp)
        tr = np.rot90(inp, k=-1)
        br = np.rot90(inp, k=-2)
        bl = np.rot90(inp, k=-3)
        
        top = np.concatenate([inp, tr], axis=1)
        bottom = np.concatenate([bl, br], axis=1)
        return np.concatenate([top, bottom], axis=0)

    for inp, out in solver.pairs:
        if not np.array_equal(apply_logic(inp), out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
