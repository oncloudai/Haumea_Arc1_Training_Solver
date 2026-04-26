
import numpy as np
from typing import List, Optional

def solve_vertical_mirror_concatenation(solver) -> Optional[List[np.ndarray]]:
    """
    Concatenates the vertically flipped input grid with the original input grid.
    Output = concatenate(vflip(inp), inp, axis=0)
    """
    def apply_logic(inp):
        inp = np.array(inp)
        vflip = inp[::-1, :]
        return np.concatenate([vflip, inp], axis=0)

    for inp, out in solver.pairs:
        if not np.array_equal(apply_logic(inp), out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
