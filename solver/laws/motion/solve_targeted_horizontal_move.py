import numpy as np
from typing import List, Optional

def solve_targeted_horizontal_move(solver) -> Optional[List[np.ndarray]]:
    for inp, out in solver.pairs:
        h, w = inp.shape; lines = {}
        for c in range(1, 10):
            for col in range(w):
                if np.sum(inp[:, col] == c) >= h * 0.7:
                    if c not in lines: lines[c] = []
                    lines[c].append(col)
        if not lines: return None
        pred = inp.copy()
        for r in range(h):
            for c in range(w):
                if inp[r,c] != 0 and (inp[r,c] not in lines or c not in lines[inp[r,c]]):
                    if inp[r,c] in lines:
                        target = min(lines[inp[r,c]], key=lambda tc: abs(tc-c)); dist = np.sign(target - c)
                        pred[r,c] = 0; pred[r, target-dist] = inp[r,c]
        if not np.array_equal(pred, out): return None
    results = []
    for ti in solver.test_in:
        h, w = ti.shape; lines = {}; res = ti.copy()
        for c in range(1, 10):
            for col in range(w):
                if np.sum(ti[:, col] == c) >= h * 0.7:
                    if c not in lines: lines[c] = []
                    lines[c].append(col)
        for r in range(h):
            for c in range(w):
                if ti[r,c] != 0 and (ti[r,c] not in lines or c not in lines[ti[r,c]]):
                    if ti[r,c] in lines:
                        res[r,c] = 0; target = min(lines[ti[r,c]], key=lambda tc: abs(tc-c)); res[r, target-np.sign(target-c)] = ti[r,c]
        results.append(res)
    return results
