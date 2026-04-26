import numpy as np
from typing import List, Optional
from scipy.ndimage import binary_fill_holes

def solve_grid_d5d6de2d(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    
    def get_components(g, color):
        visited = np.zeros_like(g, dtype=bool)
        components = []
        for r in range(h):
            for c in range(w):
                if not visited[r, c] and g[r, c] == color:
                    comp = []
                    stack = [(r, c)]
                    visited[r, c] = True
                    while stack:
                        curr_r, curr_c = stack.pop()
                        comp.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and g[nr, nc] == color:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                    components.append(comp)
        return components

    output_grid = np.zeros_like(grid)
    comps = get_components(grid, 2)
    for comp in comps:
        r_coords = [p[0] for p in comp]
        c_coords = [p[1] for p in comp]
        r_min, r_max = min(r_coords), max(r_coords)
        c_min, c_max = min(c_coords), max(c_coords)
        mask = np.zeros((r_max - r_min + 3, c_max - c_min + 3), dtype=bool)
        for r, c in comp: mask[r - r_min + 1, c - c_min + 1] = True
        filled = binary_fill_holes(mask)
        holes = filled & ~mask
        for r_local in range(holes.shape[0]):
            for c_local in range(holes.shape[1]):
                if holes[r_local, c_local]:
                    r_global = r_local + r_min - 1
                    c_global = c_local + c_min - 1
                    if 0 <= r_global < h and 0 <= c_global < w: output_grid[r_global, c_global] = 3
    return output_grid

def solve_fill_enclosed_holes_with_color_3(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_d5d6de2d(inp)
        if not np.array_equal(res, out):
            consistent = False; break
    if consistent:
        return [solve_grid_d5d6de2d(ti) for ti in solver.test_in]
    return None
