import numpy as np
from typing import List, Optional

def solve_grid_ae3edfdc(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    unique_colors, counts = np.unique(grid, return_counts=True)
    anchors = []
    particles = []
    for c, count in zip(unique_colors, counts):
        if c == 0: continue
        if count == 1: anchors.append(c)
        else: particles.append(c)
    output_grid = grid.copy()
    for p_color in particles:
        rows, cols = np.where(grid == p_color)
        for r, c in zip(rows, cols): output_grid[r, c] = 0
    for a_color in anchors:
        ar_indices, ac_indices = np.where(grid == a_color)
        if len(ar_indices) == 0: continue
        ar, ac = ar_indices[0], ac_indices[0]
        for p_color in particles:
            rows, cols = np.where(grid == p_color)
            for pr, pc in zip(rows, cols):
                if pc == ac:
                    if pr < ar and ar - 1 >= 0: output_grid[ar - 1, ac] = p_color
                    elif pr > ar and ar + 1 < h: output_grid[ar + 1, ac] = p_color
                elif pr == ar:
                    if pc < ac and ac - 1 >= 0: output_grid[ar, ac - 1] = p_color
                    elif pc > ac and ac + 1 < w: output_grid[ar, ac + 1] = p_color
    return output_grid

def solve_draw_cross_at_anchor_by_particles(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_ae3edfdc(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_ae3edfdc(ti) for ti in solver.test_in]
    return None
