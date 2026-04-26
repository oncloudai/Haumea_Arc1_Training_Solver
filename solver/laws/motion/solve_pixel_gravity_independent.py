import numpy as np
from typing import List, Optional

def solve_pixel_gravity_independent(solver) -> Optional[List[np.ndarray]]:
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for bg in range(10):
            consistent = True; found_move = False
            for inp, out in solver.pairs:
                h, w = inp.shape; res = np.full_like(inp, bg); coords = np.argwhere(inp != bg)
                if dr == 1: coords = coords[coords[:,0].argsort()[::-1]]
                elif dr == -1: coords = coords[coords[:,0].argsort()]
                elif dc == 1: coords = coords[coords[:,1].argsort()[::-1]]
                elif dc == -1: coords = coords[coords[:,1].argsort()]
                for r, c in coords:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and res[nr, nc] == bg: res[nr, nc] = inp[r, c]; found_move = True
                    else: res[r, c] = inp[r, c]
                if not np.array_equal(res, out): consistent = False; break
            if consistent and found_move:
                results = []
                for ti in solver.test_in:
                    h, w = ti.shape; res = np.full_like(ti, bg); coords = np.argwhere(ti != bg)
                    if dr == 1: coords = coords[coords[:,0].argsort()[::-1]]
                    elif dr == -1: coords = coords[coords[:,0].argsort()]
                    elif dc == 1: coords = coords[coords[:,1].argsort()[::-1]]
                    elif dc == -1: coords = coords[coords[:,1].argsort()]
                    for r, c in coords:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and res[nr, nc] == bg: res[nr, nc] = ti[r, c]
                        else: res[r, c] = ti[r, c]
                    results.append(res)
                return results
    return None
