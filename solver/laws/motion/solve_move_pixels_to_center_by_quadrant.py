import numpy as np
from typing import List, Optional

def solve_grid_d89b689b(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    unique_colors = [c for c in np.unique(grid) if c != 0]
    marker_pixels = np.argwhere(grid == 8)
    if len(marker_pixels) == 0: return grid
    mr, mc = np.mean(marker_pixels[:, 0]), np.mean(marker_pixels[:, 1])
    output_grid = np.zeros_like(grid)
    target_tl = (int(mr - 0.5), int(mc - 0.5))
    target_tr = (int(mr - 0.5), int(mc + 0.5))
    target_bl = (int(mr + 0.5), int(mc - 0.5))
    target_br = (int(mr + 0.5), int(mc + 0.5))
    for color in unique_colors:
        if color == 8: continue
        pixels = np.argwhere(grid == color)
        for r, c in pixels:
            if r < mr:
                if c < mc: tr, tc = target_tl
                else: tr, tc = target_tr
            else:
                if c < mc: tr, tc = target_bl
                else: tr, tc = target_br
            if 0 <= tr < h and 0 <= tc < w: output_grid[tr, tc] = color
    # Add marker back? Original logic doesn't, but let's see. 
    # Example outputs show ONLY the moved pixels.
    return output_grid

def solve_move_pixels_to_center_by_quadrant(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_d89b689b(inp)
        if not np.array_equal(res, out):
            consistent = False; break
    if consistent:
        return [solve_grid_d89b689b(ti) for ti in solver.test_in]
    return None
