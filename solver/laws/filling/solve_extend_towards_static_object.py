import numpy as np
from typing import List, Optional

def solve_grid_d43fd935(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    c3_pixels = np.argwhere(grid == 3)
    if len(c3_pixels) == 0: return grid
    r_min3, r_max3 = c3_pixels[:, 0].min(), c3_pixels[:, 0].max()
    c_min3, c_max3 = c3_pixels[:, 1].min(), c3_pixels[:, 1].max()
    output_grid = grid.copy()
    unique_colors = [c for c in np.unique(grid) if c != 0 and c != 3]
    for color in unique_colors:
        pixels = np.argwhere(grid == color)
        for r, c in pixels:
            if r_min3 <= r <= r_max3:
                if c > c_max3:
                    for nc in range(c_max3 + 1, c): output_grid[r, nc] = color
                elif c < c_min3:
                    for nc in range(c + 1, c_min3): output_grid[r, nc] = color
            elif c_min3 <= c <= c_max3:
                if r > r_max3:
                    for nr in range(r_max3 + 1, r): output_grid[nr, c] = color
                elif r < r_min3:
                    for nr in range(r + 1, r_min3): output_grid[nr, c] = color
    return output_grid

def solve_extend_towards_static_object(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_d43fd935(inp)
        if not np.array_equal(res, out):
            consistent = False; break
    if consistent:
        return [solve_grid_d43fd935(ti) for ti in solver.test_in]
    return None
