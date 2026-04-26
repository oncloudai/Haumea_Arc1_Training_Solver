import numpy as np
from typing import List, Optional

def solve_grid_aabf363d(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    unique_colors, counts = np.unique(grid, return_counts=True)
    shape_color = -1
    target_color = -1
    for c, count in zip(unique_colors, counts):
        if c == 0: continue
        if count == 1: target_color = c
        elif count > 1: shape_color = c
    if shape_color == -1 or target_color == -1: return grid
    output_grid = np.zeros_like(grid)
    for r in range(h):
        for c in range(w):
            if grid[r, c] == shape_color: output_grid[r, c] = target_color
            elif grid[r, c] == target_color: output_grid[r, c] = 0
            else: output_grid[r, c] = grid[r, c]
    return output_grid

def solve_recolor_shape_to_target_color(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_aabf363d(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_aabf363d(ti) for ti in solver.test_in]
    return None
