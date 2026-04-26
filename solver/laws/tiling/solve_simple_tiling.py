import numpy as np
from typing import List, Optional

def solve_simple_tiling(solver) -> Optional[List[np.ndarray]]:
    for fh in [1, 2, 3]:
        for fw in [1, 2, 3]:
            if fh == 1 and fw == 1: continue
            consistent = True
            for inp, out in solver.pairs:
                tiled = np.tile(inp, (fh, fw))
                if not np.array_equal(tiled, out): consistent = False; break
            if consistent:
                return [np.tile(ti, (fh, fw)) for ti in solver.test_in]
    return None
