import numpy as np
from typing import List, Optional

def solve_kronecker(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        try:
            k = np.kron(inp > 0, inp)
            if k.shape != out.shape or not np.array_equal(k, out): consistent = False; break
        except: consistent = False; break
    if consistent: return [np.kron(ti > 0, ti) for ti in solver.test_in]
    return None
