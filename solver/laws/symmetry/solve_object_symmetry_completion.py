import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_object_symmetry_completion(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        for conn in [4, 8]:
            consistent = True; found_any_change = False
            for inp, out in solver.pairs:
                if inp.shape != out.shape: consistent = False; break
                blobs = get_blobs(inp, bg, conn); pred = inp.copy()
                for b in blobs:
                    coords = b['coords']
                    r1, r2 = coords[:, 0].min(), coords[:, 0].max(); c1, c2 = coords[:, 1].min(), coords[:, 1].max()
                    cr, cc = (r1 + r2) / 2.0, (c1 + c2) / 2.0
                    for _ in range(3):
                        changed = False
                        for r, c in coords: # Original coords
                            for nr, nc in [(r, 2*cc-c), (2*cr-r, c), (2*cr-r, 2*cc-c), (cr+(c-cc), cc+(r-cr)), (cr-(c-cc), cc-(r-cr))]:
                                ir, ic = int(round(float(nr))), int(round(float(nc)))
                                if 0 <= ir < inp.shape[0] and 0 <= ic < inp.shape[1] and pred[ir, ic] == bg:
                                    pred[ir, ic] = b['color']; found_any_change = True
                if not np.array_equal(pred, out): consistent = False; break
            if consistent and found_any_change:
                results = []
                for ti in solver.test_in:
                    res = ti.copy(); blobs = get_blobs(ti, bg, conn)
                    for b in blobs:
                        coords = b['coords']; r1, r2 = coords[:, 0].min(), coords[:, 0].max(); c1, c2 = coords[:, 1].min(), coords[:, 1].max(); cr, cc = (r1 + r2) / 2.0, (c1 + c2) / 2.0
                        for r, c in coords:
                            for nr, nc in [(r, 2*cc-c), (2*cr-r, c), (2*cr-r, 2*cc-c), (cr+(c-cc), cc+(r-cr)), (cr-(c-cc), cc-(r-cr))]:
                                ir, ic = int(round(float(nr))), int(round(float(nc)))
                                if 0 <= ir < ti.shape[0] and 0 <= ic < ti.shape[1] and res[ir, ic] == bg: res[ir, ic] = b['color']
                    results.append(res)
                return results
    return None
