import numpy as np
from typing import List, Optional
from collections import deque

def solve_fill_reachable_zeros_from_color_1(solver) -> Optional[List[np.ndarray]]:
    """
    Logic: Seeded Background Connectivity.
    1. Identify all pixels with color 1 (blue) as seeds.
    2. Perform a 4-connected flood fill from these seeds into color 0 (background).
    3. Any 0 reached by the fill becomes 1.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        out = grid.copy()
        
        # Initialize queue with all blue seeds (color 1)
        queue = deque()
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 1:
                    queue.append((r, c))
        
        visited = set(queue)
        
        # 4-connected flood fill (Up, Down, Left, Right)
        while queue:
            r, c = queue.popleft()
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < rows and 0 <= nc < cols:
                    # Fill only background (0) pixels reachable from seeds
                    if grid[nr, nc] == 0 and (nr, nc) not in visited:
                        out[nr, nc] = 1
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return out

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
