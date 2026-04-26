import numpy as np
from typing import List, Optional

def solve_object_gravity_with_wrap(solver) -> Optional[List[np.ndarray]]:
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        # Only try background colors that are actually present
        all_colors = np.unique(solver.train_in[0])
        for bg in all_colors:
            consistent = True; found_any_move = False
            for inp, out in solver.pairs:
                res = inp.copy(); h, w = res.shape; moved = True; iters = 0
                while moved and iters < h * w:
                    moved = False; iters += 1; coords = np.argwhere(res != bg)
                    if dr == 1: coords = coords[coords[:,0].argsort()[::-1]]
                    elif dr == -1: coords = coords[coords[:,0].argsort()]
                    elif dc == 1: coords = coords[coords[:,1].argsort()[::-1]]
                    elif dc == -1: coords = coords[coords[:,1].argsort()]
                    for r, c in coords:
                        nr, nc = (r + dr) % h, (c + dc) % w
                        if res[nr, nc] == bg: res[nr, nc] = res[r, c]; res[r, c] = bg; moved = True; found_any_move = True
                if not np.array_equal(res, out): consistent = False; break
            if consistent and found_any_move:
                results = []
                for ti in solver.test_in:
                    res = ti.copy(); h, w = res.shape; moved = True; iters = 0
                    while moved and iters < h * w:
                        moved = False; iters += 1; coords = np.argwhere(res != bg)
                        if dr == 1: coords = coords[coords[:,0].argsort()[::-1]]
                        elif dr == -1: coords = coords[coords[:,0].argsort()]
                        elif dc == 1: coords = coords[coords[:,1].argsort()[::-1]]
                        elif dc == -1: coords = coords[coords[:,1].argsort()]
                        for r, c in coords:
                            nr, nc = (r + dr) % h, (c + dc) % w
                            if res[nr, nc] == bg: res[nr, nc] = res[r, c]; res[r, c] = bg; moved = True
                    results.append(res)
                return results
    return None
