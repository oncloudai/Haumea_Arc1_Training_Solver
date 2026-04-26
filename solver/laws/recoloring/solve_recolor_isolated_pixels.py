import numpy as np
from typing import List, Optional

def solve_grid_aedd82e4(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    output_grid = grid.copy()
    
    def get_connected_components(g, color):
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

    components = get_connected_components(grid, 2)
    for comp in components:
        if len(comp) == 1:
            r, c = comp[0]
            output_grid[r, c] = 1
            
    return output_grid

def solve_recolor_isolated_pixels(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_aedd82e4(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_aedd82e4(ti) for ti in solver.test_in]
    return None
