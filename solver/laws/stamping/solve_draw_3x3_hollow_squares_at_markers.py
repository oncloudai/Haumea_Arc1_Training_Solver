import numpy as np
from typing import List, Optional

def solve_grid_dc1df850(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    output_grid = grid.copy()
    c2_pixels = np.argwhere(grid == 2)
    for r, c in c2_pixels:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if output_grid[nr, nc] == 0: output_grid[nr, nc] = 1
    return output_grid

def solve_draw_3x3_hollow_squares_at_markers(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_dc1df850(inp)
        if not np.array_equal(res, out):
            consistent = False; break
    if consistent:
        return [solve_grid_dc1df850(ti) for ti in solver.test_in]
    return None
