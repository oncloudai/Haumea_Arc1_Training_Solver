import numpy as np
from typing import List, Optional

def solve_grid_dbc1a6ce(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    output_grid = grid.copy()
    pixels = np.argwhere(grid == 1)
    for i in range(len(pixels)):
        r1, c1 = pixels[i]
        min_dist_c = float('inf'); target_c = -1
        for j in range(len(pixels)):
            if i == j: continue
            r2, c2 = pixels[j]
            if r1 == r2 and c2 > c1:
                dist = c2 - c1
                if dist < min_dist_c: min_dist_c = dist; target_c = c2
        if target_c != -1:
            for c in range(c1 + 1, target_c): output_grid[r1, c] = 8
        min_dist_r = float('inf'); target_r = -1
        for j in range(len(pixels)):
            if i == j: continue
            r2, c2 = pixels[j]
            if c1 == c2 and r2 > r1:
                dist = r2 - r1
                if dist < min_dist_r: min_dist_r = dist; target_r = r2
        if target_r != -1:
            for r in range(r1 + 1, target_r): output_grid[r, c1] = 8
    return output_grid

def solve_connect_same_color_orthogonal_with_8(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_dbc1a6ce(inp)
        if not np.array_equal(res, out):
            consistent = False; break
    if consistent:
        return [solve_grid_dbc1a6ce(ti) for ti in solver.test_in]
    return None
