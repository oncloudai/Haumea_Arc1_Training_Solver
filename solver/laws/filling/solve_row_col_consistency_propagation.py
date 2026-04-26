import numpy as np
from typing import List, Optional

def solve_row_col_consistency_propagation(solver) -> Optional[List[np.ndarray]]:
    def propagate(grid):
        res = grid.copy(); h, w = res.shape; changed = True
        while changed:
            changed = False
            for i in range(h):
                for j in range(h):
                    if i == j: continue
                    if all((res[i,c]==0 or res[j,c]==0 or res[i,c]==res[j,c]) for c in range(w)):
                        for c in range(w):
                            if res[i,c] == 0 and res[j,c] != 0: res[i,c] = res[j,c]; changed = True
            for i in range(w):
                for j in range(w):
                    if i == j: continue
                    if all((res[r,i]==0 or res[r,j]==0 or res[r,i]==res[r,j]) for r in range(h)):
                        for r in range(h):
                            if res[r,i] == 0 and res[r,j] != 0: res[r,i] = res[r,j]; changed = True
        return res
    for inp, out in solver.pairs:
        if inp.shape != out.shape: return None
        if not np.array_equal(propagate(inp), out): return None
    return [propagate(ti) for ti in solver.test_in]
