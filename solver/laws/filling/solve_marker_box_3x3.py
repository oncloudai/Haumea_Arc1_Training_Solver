import numpy as np
from typing import List, Optional

def solve_marker_box_3x3(solver) -> Optional[List[np.ndarray]]:
    box_color = 2
    for inp, out in solver.pairs:
        h, w = inp.shape; unq, counts = np.unique(inp, return_counts=True); singletons = unq[counts == 1]
        pred = np.zeros_like(out); found_any = False
        for color in singletons:
            if color == 0: continue
            r, c = np.argwhere(inp == color)[0]
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w: pred[nr, nc] = box_color; found_any = True
            if 0 <= r < h and 0 <= c < w: pred[r, c] = color
        if not np.array_equal(pred, out): return None
    results = []
    for ti in solver.test_in:
        h, w = ti.shape; unq, counts = np.unique(ti, return_counts=True); singletons = unq[counts == 1]; res = np.zeros((h, w), dtype=int)
        for color in singletons:
            if color == 0: continue
            r, c = np.argwhere(ti == color)[0]
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w: res[nr, nc] = box_color
            res[r, c] = color
        results.append(res)
    return results
