import numpy as np
from typing import List, Optional

def solve_3x3_periodic_recolor(solver) -> Optional[List[np.ndarray]]:
    for c1 in range(1, 10):
        for c2 in range(1, 10):
            if c1 == c2: continue
            base = np.array([[c1, c1, c2], [c1, c1, c2], [c2, c2, c2]])
            for dr in range(3):
                for dc in range(3):
                    consistent = True; results = []
                    for inp, out in solver.pairs:
                        if inp.shape != out.shape: consistent = False; break
                        pred = inp.copy()
                        for r in range(inp.shape[0]):
                            for c in range(inp.shape[1]):
                                if inp[r, c] != 0: pred[r, c] = base[(r+dr)%3, (c+dc)%3]
                        if not np.array_equal(pred, out): consistent = False; break
                    if consistent:
                        for ti in solver.test_in:
                            res = ti.copy()
                            for r in range(ti.shape[0]):
                                for c in range(ti.shape[1]):
                                    if ti[r, c] != 0: res[r, c] = base[(r+dr)%3, (c+dc)%3]
                            results.append(res)
                        return results
    return None
