import numpy as np
from typing import List, Optional

def solve_recolor_components_by_path_topology(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies connected components of color 3.
    Classifies each component based on its path topology (junctions, turns, loops).
    - Junctions (degree > 2): Red (2)
    - Loops or many turns: Pink (6)
    - Simple paths (turns <= 1): Blue (1)
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

    def classify_component(comp):
        pixels = set(comp)
        has_junction = False
        endpoints = []
        for r, c in comp:
            neighbors = 0
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if (r + dr, c + dc) in pixels: neighbors += 1
            if neighbors > 2:
                has_junction = True; break
            if neighbors == 1: endpoints.append((r, c))
                
        if has_junction: return 2 # Red
        if not endpoints: return 6 # Pink (Loop)
            
        curr = endpoints[0]
        visited = {curr}
        path = [curr]
        while True:
            next_pixel = None
            r, c = curr
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (r + dr, c + dc)
                if neighbor in pixels and neighbor not in visited:
                    next_pixel = neighbor; break
            if next_pixel is None: break
            visited.add(next_pixel)
            path.append(next_pixel)
            curr = next_pixel
            
        if len(path) < 3: return 1
            
        turns = 0
        prev_dr, prev_dc = path[1][0] - path[0][0], path[1][1] - path[0][1]
        for i in range(2, len(path)):
            dr, dc = path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]
            if dr != prev_dr or dc != prev_dc:
                turns += 1
                prev_dr, prev_dc = dr, dc
                
        if turns <= 1: return 1 # Blue
        else: return 6 # Pink

    def apply_logic(grid):
        grid = np.array(grid)
        out = grid.copy()
        # Find potential color to replace (usually 3 in this task)
        # But for general law, we should find which color is being replaced.
        # We'll stick to 3 if it's the only one, or try to infer.
        unique = np.unique(grid)
        if 3 not in unique: return grid
        
        comps = get_components(grid, 3)
        for comp in comps:
            color = classify_component(comp)
            for r, c in comp: out[r, c] = color
        return out

    for inp, out in solver.pairs:
        if not np.array_equal(apply_logic(inp), out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
