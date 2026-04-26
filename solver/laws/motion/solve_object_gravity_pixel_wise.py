import numpy as np
from typing import List, Optional

def solve_object_gravity_pixel_wise(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            consistent = True; found_move = False
            for inp, out in solver.pairs:
                if inp.shape != out.shape: consistent = False; break
                h, w = inp.shape; res = inp.copy()
                for _ in range(max(h, w)):
                    moved_any = False; r_range = range(h-2, -1, -1) if dr == 1 else range(1, h) if dr == -1 else range(h)
                    c_range = range(w-2, -1, -1) if dc == 1 else range(1, w) if dc == -1 else range(w)
                    for r in r_range:
                        for c in c_range:
                            if res[r, c] != bg and res[r+dr, c+dc] == bg:
                                res[r+dr, c+dc] = res[r, c]; res[r, c] = bg; moved_any = True; found_move = True
                    if not moved_any: break
                if not np.array_equal(res, out): consistent = False; break
            if consistent and found_move:
                results = []
                for ti in solver.test_in:
                    h, w = ti.shape; res = ti.copy()
                    for _ in range(max(h, w)):
                        moved_any = False; r_range = range(h-2, -1, -1) if dr == 1 else range(1, h) if dr == -1 else range(h)
                        c_range = range(w-2, -1, -1) if dc == 1 else range(1, w) if dc == -1 else range(w)
                        for r in r_range:
                            for c in c_range:
                                if res[r, c] != bg and res[r+dr, c+dc] == bg:
                                    res[r+dr, c+dc] = res[r, c]; res[r, c] = bg; moved_any = True
                        if not moved_any: break
                    results.append(res)
                return results
    return None
