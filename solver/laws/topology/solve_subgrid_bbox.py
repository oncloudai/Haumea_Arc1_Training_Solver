import numpy as np
from typing import List, Optional

def solve_subgrid_bbox(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        consistent = True; results = []
        for inp, out in solver.pairs:
            non_bg = np.argwhere(inp != bg)
            if len(non_bg) == 0: consistent = False; break
            r1, c1 = non_bg.min(axis=0); r2, c2 = non_bg.max(axis=0)
            sub = inp[r1:r2+1, c1:c2+1]
            if not np.array_equal(sub, out): consistent = False; break
        if consistent:
            for ti in solver.test_in:
                non_bg = np.argwhere(ti != bg)
                if len(non_bg) == 0: break
                r1, c1 = non_bg.min(axis=0); r2, c2 = non_bg.max(axis=0)
                results.append(ti[r1:r2+1, c1:c2+1])
            if len(results) == len(solver.test_in): return results
    return None
