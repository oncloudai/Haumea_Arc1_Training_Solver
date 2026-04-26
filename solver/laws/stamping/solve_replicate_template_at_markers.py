import numpy as np
from typing import List, Optional

def solve_replicate_template_at_markers(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a template (all pixels that are not 0 and not 5).
    Identifies 'markers' (all connected components of color 5).
    For each marker, copies the template starting at the marker's top-left position.
    Original 5s are removed.
    """
    def get_components(grid, color):
        h, w = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        components = []
        for r in range(h):
            for c in range(w):
                if not visited[r, c] and grid[r, c] == color:
                    comp = []
                    stack = [(r, c)]
                    visited[r, c] = True
                    while stack:
                        curr_r, curr_c = stack.pop()
                        comp.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == color:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                    components.append(comp)
        return components

    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # 1. Identify the template
        template_mask = (grid != 0) & (grid != 5)
        template_pixels = np.argwhere(template_mask)
        if len(template_pixels) == 0: return grid
            
        r_min, r_max = np.min(template_pixels[:, 0]), np.max(template_pixels[:, 0])
        c_min, c_max = np.min(template_pixels[:, 1]), np.max(template_pixels[:, 1])
        h_t, w_t = r_max - r_min + 1, c_max - c_min + 1
        
        template_subgrid = grid[r_min:r_max+1, c_min:c_max+1].copy()
        template_subgrid[template_subgrid == 5] = 0
        
        # 2. Find all color 5 components
        comps_5 = get_components(grid, 5)
        
        output = grid.copy()
        output[output == 5] = 0
        
        # 3. For each color 5 component, copy the template
        for comp in comps_5:
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r5_min, c5_min = min(rs), min(cs)
            
            for r in range(h_t):
                for c in range(w_t):
                    if template_subgrid[r, c] != 0:
                        tr, tc = r5_min + r, c5_min + c
                        if 0 <= tr < h and 0 <= tc < w:
                            output[tr, tc] = template_subgrid[r, c]
        return output

    for inp, out in solver.pairs:
        if not np.array_equal(apply_logic(inp), out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
