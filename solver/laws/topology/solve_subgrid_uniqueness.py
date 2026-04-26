import numpy as np
from typing import List, Optional

def solve_subgrid_uniqueness(solver) -> Optional[List[np.ndarray]]:
    for d in range(1, 10):
        consistent = True; results = []
        for inp, out in solver.pairs:
            rows = np.where(np.all(inp == d, axis=1))[0]; cols = np.where(np.all(inp == d, axis=0))[0]
            if len(rows) < 1 or len(cols) < 1: consistent = False; break
            rb = [-1] + sorted(list(rows)) + [inp.shape[0]]; cb = [-1] + sorted(list(cols)) + [inp.shape[1]]
            cells = []
            for r in range(len(rb)-1):
                for c in range(len(cb)-1):
                    cells.append(inp[rb[r]+1:rb[r+1], cb[c]+1:cb[c+1]])
            unq = [c for c in cells if sum(np.array_equal(c, other) for other in cells) == 1]
            if len(unq) != 1 or not np.array_equal(unq[0], out): consistent = False; break
        if consistent:
            for ti in solver.test_in:
                rows = np.where(np.all(ti == d, axis=1))[0]; cols = np.where(np.all(ti == d, axis=0))[0]
                rb = [-1] + sorted(list(rows)) + [ti.shape[0]]; cb = [-1] + sorted(list(cols)) + [ti.shape[1]]
                cells = [ti[rb[r]+1:rb[r+1], cb[c]+1:cb[c+1]] for r in range(len(rb)-1) for c in range(len(cb)-1)]
                unq = [c for c in cells if sum(np.array_equal(c, other) for other in cells) == 1]
                if len(unq) == 1: results.append(unq[0])
                else: break
            if len(results) == len(solver.test_in): return results
    return None
