import numpy as np
from typing import List, Optional

def solve_grid_d8c310e9(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    for p in range(1, w):
        possible = True
        for c in range(w):
            source_c = c % p
            for r in range(h):
                val, source_val = grid[r, c], grid[r, source_c]
                if val != 0 and source_val != 0 and val != source_val:
                    possible = False; break
            if not possible: break
        if possible:
            output_grid = np.zeros_like(grid)
            seed = np.zeros((h, p), dtype=int)
            for r in range(h):
                for c in range(w):
                    if grid[r, c] != 0: seed[r, c % p] = grid[r, c]
            for r in range(h):
                for c in range(w): output_grid[r, c] = seed[r, c % p]
            return output_grid
    return grid

def solve_horizontal_tiling_by_period(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_d8c310e9(inp)
        if not np.array_equal(res, out):
            consistent = False; break
    if consistent:
        return [solve_grid_d8c310e9(ti) for ti in solver.test_in]
    return None
