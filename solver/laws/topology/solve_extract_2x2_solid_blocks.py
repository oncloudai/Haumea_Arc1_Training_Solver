import numpy as np
from typing import List, Optional

def solve_grid_7f4411dc(grid):
    grid = np.array(grid)
    h, w = grid.shape
    out = np.zeros_like(grid)
    for r in range(h - 1):
        for c in range(w - 1):
            color = grid[r, c]
            if color > 0:
                if (grid[r+1, c] == color and grid[r, c+1] == color and grid[r+1, c+1] == color):
                    out[r, c] = color; out[r+1, c] = color; out[r, c+1] = color; out[r+1, c+1] = color
    return out

def solve_extract_2x2_solid_blocks(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_7f4411dc(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_7f4411dc(ti) for ti in solver.test_in]
    return None
