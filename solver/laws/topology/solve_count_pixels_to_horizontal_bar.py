import numpy as np
from typing import List, Optional

def solve_grid_d631b094(input_grid):
    grid = np.array(input_grid)
    unique_colors = [c for c in np.unique(grid) if c != 0]
    if not unique_colors: return np.zeros((1, 1), dtype=int)
    color = unique_colors[0]
    count = np.sum(grid == color)
    output_grid = np.full((1, int(count)), color, dtype=int)
    return output_grid

def solve_count_pixels_to_horizontal_bar(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_d631b094(inp)
        if not np.array_equal(res, out):
            consistent = False; break
    if consistent:
        return [solve_grid_d631b094(ti) for ti in solver.test_in]
    return None
