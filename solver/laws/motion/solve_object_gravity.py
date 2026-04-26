import numpy as np
from typing import List, Optional

def solve_object_gravity(solver) -> Optional[List[np.ndarray]]:
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        all_colors = np.unique(solver.train_in[0])
        for bg in all_colors:
            consistent = True; found_move = False
            for inp, out in solver.pairs:
                res = inp.copy(); moved = True
                while moved:
                    moved = False; coords = np.argwhere(res != bg)
                    if dr == 1: coords = coords[coords[:,0].argsort()[::-1]]
                    elif dr == -1: coords = coords[coords[:,0].argsort()]
                    elif dc == 1: coords = coords[coords[:,1].argsort()[::-1]]
                    elif dc == -1: coords = coords[coords[:,1].argsort()]
                    for r, c in coords:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < res.shape[0] and 0 <= nc < res.shape[1] and res[nr, nc] == bg:
                            res[nr, nc] = res[r, c]; res[r, c] = bg; moved = True; found_move = True
                if not np.array_equal(res, out): consistent = False; break
            if consistent and found_move:
                results = []
                for ti in solver.test_in:
                    res = ti.copy(); moved = True
                    while moved:
                        moved = False; coords = np.argwhere(res != bg)
                        if dr == 1: coords = coords[coords[:,0].argsort()[::-1]]
                        elif dr == -1: coords = coords[coords[:,0].argsort()]
                        elif dc == 1: coords = coords[coords[:,1].argsort()[::-1]]
                        elif dc == -1: coords = coords[coords[:,1].argsort()]
                        for r, c in coords:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < res.shape[0] and 0 <= nc < res.shape[1] and res[nr, nc] == bg:
                                res[nr, nc] = res[r, c]; res[r, c] = bg; moved = True
                    results.append(res)
                return results
    return None
