import numpy as np
from typing import List, Optional

def solve_grid_ddf7fa4f(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    def get_components(g, color):
        visited = np.zeros_like(g, dtype=bool)
        components = []
        for r in range(h):
            for c in range(w):
                if not visited[r, c] and g[r, c] == color:
                    comp = []; stack = [(r, c)]; visited[r, c] = True
                    while stack:
                        curr_r, curr_c = stack.pop(); comp.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and g[nr, nc] == color:
                                visited[nr, nc] = True; stack.append((nr, nc))
                    components.append(comp)
        return components

    pieces = get_components(grid, 5)
    output_grid = np.zeros_like(grid)
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0 and grid[r, c] != 5: output_grid[r, c] = grid[r, c]
    markers = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0 and grid[r, c] != 5: markers.append({'color': grid[r, c], 'r': r, 'c': c})
    for piece in pieces:
        comp_pixels = np.array(piece)
        c_min, c_max = comp_pixels[:, 1].min(), comp_pixels[:, 1].max()
        found_marker = None
        for m in markers:
            if c_min <= m['c'] <= c_max: found_marker = m; break
        if found_marker:
            for r, c in piece: output_grid[r, c] = found_marker['color']
    return output_grid

def solve_recolor_pieces_by_column_marker(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_ddf7fa4f(inp)
        if not np.array_equal(res, out):
            consistent = False; break
    if consistent:
        return [solve_grid_ddf7fa4f(ti) for ti in solver.test_in]
    return None
