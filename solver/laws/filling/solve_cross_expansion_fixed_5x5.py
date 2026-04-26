import numpy as np
from typing import List, Optional

def solve_cross_expansion_fixed_5x5(solver) -> Optional[List[np.ndarray]]:
    bg = 0
    # Redefine process more precisely
    def process_precise(grid):
        res = grid.copy(); h, w = grid.shape
        offsets_C = [(0,0), (1,1), (1,-1), (-1,1), (-1,-1), (2,2), (2,-2), (-2,2), (-2,-2)]
        offsets_A = [(0,1), (0,-1), (1,0), (-1,0), (0,2), (0,-2), (2,0), (-2,0)]
        for r in range(h):
            for c in range(w):
                # Potential center of a 3x3 cross
                if r < 1 or r >= h-1 or c < 1 or c >= w-1: continue
                C, A = grid[r, c], grid[r-1, c]
                if C == bg or A == bg or A == C: continue
                if all(grid[r+dr, c+dc] == A for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]):
                    if all(grid[r+dr, c+dc] == bg for dr, dc in [(1,1), (1,-1), (-1,1), (-1,-1)]):
                        # Apply 5x5 expansion
                        for dr, dc in offsets_C:
                            if 0 <= r+dr < h and 0 <= c+dc < w: res[r+dr, c+dc] = C
                        for dr, dc in offsets_A:
                            if 0 <= r+dr < h and 0 <= c+dc < w: res[r+dr, c+dc] = A
        return res
    for inp, out in solver.pairs:
        if not np.array_equal(process_precise(inp), out): return None
    return [process_precise(ti) for ti in solver.test_in]
