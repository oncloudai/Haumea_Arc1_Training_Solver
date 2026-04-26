import numpy as np
from typing import List, Optional

def solve_mirror_pattern_by_2x2_junction(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies 2x2 junctions where multiple unique colors are present.
    For each junction, finds patterns (8-connected components) attached to each color.
    Mirrors patterns from one color to another within the junction:
    - If seeds are in same row: horizontal mirror.
    - If seeds are in same column: vertical mirror.
    - If seeds are diagonal: 180-degree rotation (double mirror).
    Repeats until stability.
    """
    def get_8_connected_component(grid, r, c):
        rows, cols = grid.shape
        color = grid[r, c]
        visited = set()
        q = [(r, c)]
        visited.add((r, c))
        while q:
            curr_r, curr_c = q.pop(0)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if grid[nr, nc] == color and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))
        return visited

    def apply_logic(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        output = grid.copy()
        
        # 1. Find all 2x2 junctions with at least 3 unique non-zero colors
        junctions = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                pixels = []
                for dr in [0, 1]:
                    for dc in [0, 1]:
                        if grid[r+dr, c+dc] != 0:
                            pixels.append((r+dr, c+dc, int(grid[r+dr, c+dc])))
                
                colors = [p[2] for p in pixels]
                if len(colors) >= 3 and len(set(colors)) == len(colors):
                    junctions.append(pixels)
                    
        if not junctions: return grid

        # 2. Propagate patterns
        for _ in range(5): # Limit passes
            changed = False
            for j_pixels in junctions:
                for s1_r, s1_c, s1_color in j_pixels:
                    p1 = get_8_connected_component(output, s1_r, s1_c)
                    if len(p1) > 1:
                        for s2_r, s2_c, s2_color in j_pixels:
                            if (s1_r, s1_c) == (s2_r, s2_c): continue
                            
                            for pr, pc in p1:
                                if s1_r == s2_r:
                                    nr, nc = pr, s1_c + s2_c - pc
                                elif s1_c == s2_c:
                                    nr, nc = s1_r + s2_r - pr, pc
                                else:
                                    nr, nc = s1_r + s2_r - pr, s1_c + s2_c - pc
                                    
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    if output[nr, nc] == 0:
                                        output[nr, nc] = s2_color
                                        changed = True
            if not changed: break
            
        return output

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
