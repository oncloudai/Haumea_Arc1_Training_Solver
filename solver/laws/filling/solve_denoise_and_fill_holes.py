import numpy as np
from typing import List, Optional

def solve_grid_7e0986d6(grid):
    grid = np.array(grid)
    h, w = grid.shape
    unique_colors = np.unique(grid)
    unique_colors = unique_colors[unique_colors > 0]
    if len(unique_colors) < 2: return grid
    counts = {c: np.sum(grid == c) for c in unique_colors}
    noise_color = min(counts, key=counts.get)
    object_color = max(counts, key=counts.get)
    out = grid.copy()
    out[out == noise_color] = 0
    for _ in range(5):
        changed = False; temp_out = out.copy()
        for r in range(h):
            for c in range(w):
                if out[r, c] == 0:
                    neighbors = []
                    if r > 0: neighbors.append(out[r-1, c])
                    if r < h-1: neighbors.append(out[r+1, c])
                    if c > 0: neighbors.append(out[r, c-1])
                    if c < w-1: neighbors.append(out[r, c+1])
                    obj_neighbors = neighbors.count(object_color)
                    if obj_neighbors >= 3: temp_out[r, c] = object_color; changed = True
                    elif obj_neighbors >= 2 and grid[r, c] == noise_color:
                         temp_out[r, c] = object_color; changed = True
        out = temp_out
        if not changed: break
    return out

def solve_denoise_and_fill_holes(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_7e0986d6(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_7e0986d6(ti) for ti in solver.test_in]
    return None
