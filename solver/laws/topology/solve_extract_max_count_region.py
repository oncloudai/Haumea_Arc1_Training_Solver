import numpy as np
from typing import List, Optional

def solve_extract_max_count_region(solver) -> Optional[List[np.ndarray]]:
    """
    Finds all connected components of non-zero pixels (4-connectivity).
    Extracts the bounding box of each component.
    Returns the component with the MAXIMUM COUNT of color-2 pixels.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # Find all connected components of non-zero pixels
        visited = np.zeros_like(grid, dtype=bool)
        regions = []
        
        for r in range(h):
            for c in range(w):
                if not visited[r, c] and grid[r, c] != 0:
                    # BFS to find connected component using 4-connectivity
                    stack = [(r, c)]
                    visited[r, c] = True
                    component = [(r, c)]
                    
                    while stack:
                        curr_r, curr_c = stack.pop()
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if (0 <= nr < h and 0 <= nc < w and 
                                not visited[nr, nc] and grid[nr, nc] != 0):
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                                component.append((nr, nc))
                    
                    rows = [p[0] for p in component]
                    cols = [p[1] for p in component]
                    r_min, r_max = min(rows), max(rows)
                    c_min, c_max = min(cols), max(cols)
                    
                    subgrid = grid[r_min:r_max+1, c_min:c_max+1]
                    count_2 = np.sum(subgrid == 2)
                    regions.append({'subgrid': subgrid, 'count_2': count_2})
        
        if not regions: return None
        best_region = max(regions, key=lambda r: r['count_2'])
        return best_region['subgrid']

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
