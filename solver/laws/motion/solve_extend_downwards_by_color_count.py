import numpy as np
from typing import List, Optional

def solve_grid_fcc82909(input_grid):
    grid = np.array(input_grid)
    rows, cols = grid.shape
    
    # 1. Identify all non-zero colors (excluding the extension color 3)
    # Actually, we should just find connected components of any color != 0
    # and not worry about color 3, but in the examples color 3 is not there.
    
    def get_components(g):
        h, w = g.shape
        visited = np.zeros((h, w), dtype=bool)
        components = []
        for r in range(h):
            for c in range(w):
                if g[r, c] != 0 and not visited[r, c]:
                    # Start new component
                    comp = []
                    stack = [(r, c)]
                    visited[r, c] = True
                    while stack:
                        curr_r, curr_c = stack.pop()
                        comp.append((curr_r, curr_c))
                        # 8-connectivity is usually better for ARC objects
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0: continue
                                nr, nc = curr_r + dr, curr_c + dc
                                if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != 0 and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    stack.append((nr, nc))
                    components.append(comp)
        return components

    components = get_components(grid)
    output = grid.copy()
    
    # We need to process components from bottom to top?
    # Or maybe it doesn't matter because original objects have priority.
    # Let's collect all extension pixels first, then apply them only if 0.
    
    extension_pixels = []
    
    for comp in components:
        # Unique colors in this component
        comp_colors = set()
        for r, c in comp:
            comp_colors.add(grid[r, c])
        num_unique = len(comp_colors)
        
        # Columns occupied
        comp_cols = set()
        for r, c in comp:
            comp_cols.add(c)
            
        for c in comp_cols:
            # Find lowest row for this column in this component
            r_max = -1
            for r_comp, c_comp in comp:
                if c_comp == c:
                    if r_comp > r_max:
                        r_max = r_comp
            
            # Extend downwards
            for r_ext in range(r_max + 1, r_max + 1 + num_unique):
                if 0 <= r_ext < rows:
                    extension_pixels.append((r_ext, c, 3))

    # Apply extensions if the cell is currently 0
    for r, c, val in extension_pixels:
        if output[r, c] == 0:
            output[r, c] = val
            
    return output

def solve_extend_downwards_by_color_count(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_fcc82909(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_fcc82909(ti) for ti in solver.test_in]
    return None
