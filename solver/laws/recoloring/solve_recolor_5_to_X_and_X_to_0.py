import numpy as np
from typing import List, Optional

def solve_grid_f76d97a5(input_grid):
    grid = np.array(input_grid)
    unique_colors = np.unique(grid)
    other_colors = [c for c in unique_colors if c != 5 and c != 0]
    if not other_colors: return grid
    X = other_colors[0]
    output_grid = np.zeros_like(grid)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 5: output_grid[r, c] = X
            elif grid[r, c] == X: output_grid[r, c] = 0
            else: output_grid[r, c] = 0
    return output_grid

def solve_recolor_5_to_X_and_X_to_0(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_f76d97a5(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_f76d97a5(ti) for ti in solver.test_in]
    return None
