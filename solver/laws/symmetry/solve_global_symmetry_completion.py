import numpy as np
from typing import List, Optional

def solve_global_symmetry_completion(solver) -> Optional[List[np.ndarray]]:
    for inp, out in solver.pairs:
        coords = np.argwhere(inp != 0)
        if len(coords) == 0: return None
        r1, r2 = coords[:, 0].min(), coords[:, 0].max(); c1, c2 = coords[:, 1].min(), coords[:, 1].max(); cr, cc = (r1 + r2) / 2.0, (c1 + c2) / 2.0; pred = inp.copy()
        for _ in range(3):
            changed = False
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    if pred[r, c] != 0:
                        for nr, nc in [(r, 2*cc-c), (2*cr-r, c), (2*cr-r, 2*cc-c), (cr+(c-cc), cc+(r-cr))]:
                            ir, ic = int(round(float(nr))), int(round(float(nc)))
                            if 0 <= ir < inp.shape[0] and 0 <= ic < inp.shape[1] and pred[ir, ic] == 0:
                                pred[ir, ic] = pred[r, c]; changed = True
            if not changed: break
        if not np.array_equal(pred, out): return None
    results = []
    for ti in solver.test_in:
        coords = np.argwhere(ti != 0); r1, r2 = coords[:, 0].min(), coords[:, 0].max(); c1, c2 = coords[:, 1].min(), coords[:, 1].max(); cr, cc = (r1 + r2) / 2.0, (c1 + c2) / 2.0; res = ti.copy()
        for _ in range(3):
            changed = False
            for r in range(ti.shape[0]):
                for c in range(ti.shape[1]):
                    if res[r, c] != 0:
                        for nr, nc in [(r, 2*cc-c), (2*cr-r, c), (2*cr-r, 2*cc-c), (cr+(c-cc), cc+(r-cr))]:
                            ir, ic = int(round(float(nr))), int(round(float(nc)))
                            if 0 <= ir < ti.shape[0] and 0 <= ic < ti.shape[1] and res[ir, ic] == 0: res[ir, ic] = res[r, c]; changed = True
            if not changed: break
        results.append(res)
    return results
