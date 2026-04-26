import numpy as np
from typing import List, Optional

def solve_grid_d6ad076f(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    def get_components(g):
        hh, ww = g.shape
        visited = np.zeros_like(g, dtype=bool)
        components = []
        for r in range(hh):
            for c in range(ww):
                if not visited[r, c] and g[r, c] != 0:
                    color = g[r, c]
                    comp = []; stack = [(r, c)]; visited[r, c] = True
                    while stack:
                        curr_r, curr_c = stack.pop(); comp.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < hh and 0 <= nc < ww and not visited[nr, nc] and g[nr, nc] == color:
                                visited[nr, nc] = True; stack.append((nr, nc))
                    rows = [p[0] for p in comp]; cols = [p[1] for p in comp]
                    components.append({'r_min': min(rows), 'r_max': max(rows), 'c_min': min(cols), 'c_max': max(cols)})
        return components

    comps = get_components(grid)
    if len(comps) != 2: return grid
    c1, c2 = comps[0], comps[1]
    r_start, r_end = max(c1['r_min'], c2['r_min']), min(c1['r_max'], c2['r_max'])
    if r_start <= r_end:
        r_f_s = r_start + (r_end - r_start + 1) // 4
        r_f_e = r_end - (r_end - r_start + 1) // 4
    else: r_f_s, r_f_e = (min(c1['r_max'], c2['r_max']) + 1, max(c1['r_min'], c2['r_min']) - 1)
    c_start, c_end = max(c1['c_min'], c2['c_min']), min(c1['c_max'], c2['c_max'])
    if c_start <= c_end:
        c_f_s = c_start + (c_end - c_start + 1) // 4
        c_f_e = c_end - (c_end - c_start + 1) // 4
    else: c_f_s, c_f_e = (min(c1['c_max'], c2['c_max']) + 1, max(c1['c_min'], c2['c_min']) - 1)
    output_grid = grid.copy()
    for r in range(int(r_f_s), int(r_f_e) + 1):
        for c in range(int(c_f_s), int(c_f_e) + 1):
            if 0 <= r < h and 0 <= c < w: output_grid[r, c] = 8
    return output_grid

def solve_fill_box_between_two_objects(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_d6ad076f(inp)
        if not np.array_equal(res, out):
            consistent = False; break
    if consistent:
        return [solve_grid_d6ad076f(ti) for ti in solver.test_in]
    return None
