import numpy as np
from typing import List, Optional

def solve_grid_6e19193c(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    output_grid = grid.copy()
    
    # 1. Connected components
    def get_components(g):
        hh, ww = g.shape
        visited = np.zeros_like(g, dtype=bool)
        components = []
        for r in range(hh):
            for c in range(ww):
                if not visited[r, c] and g[r, c] != 0:
                    color = g[r, c]
                    comp = []
                    stack = [(r, c)]
                    visited[r, c] = True
                    while stack:
                        curr_r, curr_c = stack.pop()
                        comp.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < hh and 0 <= nc < ww and not visited[nr, nc] and g[nr, nc] == color:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                    components.append({'color': color, 'pixels': comp})
        return components

    comps = get_components(grid)
    for comp in comps:
        pixels = comp['pixels']
        if len(pixels) != 3: continue
        
        # Identify L-shape arrowhead
        joint = None
        tips = []
        for p in pixels:
            count = 0
            curr_tips = []
            for other in pixels:
                if p == other: continue
                # 4-neighbors only for arrowhead joint
                if abs(p[0]-other[0]) + abs(p[1]-other[1]) == 1:
                    count += 1
                    curr_tips.append(other)
            if count == 2:
                joint = p
                tips = curr_tips
                break
        
        if joint:
            # Inside corner C = T1 + T2 - J
            c_r = tips[0][0] + tips[1][0] - joint[0]
            c_c = tips[0][1] + tips[1][1] - joint[1]
            
            # Direction d = C - J
            dr, dc = c_r - joint[0], c_c - joint[1]
            if dr == 0 and dc == 0: continue
            
            # Draw ray C + k*d for k=1, 2, ...
            curr_r, curr_c = c_r + dr, c_c + dc
            while 0 <= curr_r < h and 0 <= curr_c < w:
                if output_grid[curr_r, curr_c] == 0:
                    output_grid[curr_r, curr_c] = comp['color']
                curr_r += dr
                curr_c += dc
                
    return output_grid

def solve_extend_l_shape_arrowheads(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_6e19193c(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_6e19193c(ti) for ti in solver.test_in]
    return None
