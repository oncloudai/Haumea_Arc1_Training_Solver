import numpy as np
from typing import List, Optional

def solve_scaling(solver) -> Optional[List[np.ndarray]]:
    for f in range(2, 6):
        consistent = True
        for inp, out in solver.pairs:
            if out.shape[0] != inp.shape[0]*f or out.shape[1] != inp.shape[1]*f: consistent = False; break
            if not np.array_equal(np.repeat(np.repeat(inp, f, axis=0), f, axis=1), out): consistent = False; break
        if consistent: return [np.repeat(np.repeat(ti, f, axis=0), f, axis=1) for ti in solver.test_in]
    return None
