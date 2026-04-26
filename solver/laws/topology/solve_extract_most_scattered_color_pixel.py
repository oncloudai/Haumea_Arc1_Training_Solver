import numpy as np
from typing import List, Optional

def solve_grid_d9fac9be(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    unique_colors = [c for c in np.unique(grid) if c != 0]
    if not unique_colors: return np.zeros((1, 1), dtype=int)
    best_color = -1; max_scatter = -1
    for color in unique_colors:
        pixels = np.argwhere(grid == color)
        num_pixels = len(pixels)
        r_min, r_max = pixels[:, 0].min(), pixels[:, 0].max()
        c_min, c_max = pixels[:, 1].min(), pixels[:, 1].max()
        bb_area = (r_max - r_min + 1) * (c_max - c_min + 1)
        scatter = bb_area / num_pixels
        if scatter > max_scatter:
            max_scatter = scatter; best_color = color
    return np.array([[int(best_color)]])

def solve_extract_most_scattered_color_pixel(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_d9fac9be(inp)
        if not np.array_equal(res, out):
            consistent = False; break
    if consistent:
        return [solve_grid_d9fac9be(ti) for ti in solver.test_in]
    return None
