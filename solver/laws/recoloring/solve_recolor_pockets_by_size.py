import numpy as np
from typing import List, Optional

def solve_recolor_pockets_by_size(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies 4-connected components of color 0 ('pockets').
    Recolors each pocket based on its size (number of pixels).
    Mapping: 1 -> 3, 2 -> 2, 3 -> 1. (Cycle: (3 - size) % 3 + 1).
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
        pockets = get_components(grid, 0)
        output = grid.copy()
        
        for pocket in pockets:
            size = len(pocket)
            color = (3 - size) % 3 + 1
            for r, c in pocket:
                output[r, c] = color
        return output

    for inp, out in solver.pairs:
        if not np.array_equal(apply_logic(inp), out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
