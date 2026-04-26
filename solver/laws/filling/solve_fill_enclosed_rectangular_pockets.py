import numpy as np
from typing import List, Optional

def solve_fill_enclosed_rectangular_pockets(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies rectangular components of color 0 and checks if they are perfectly 
    enclosed by lines of color 5 or grid boundaries. 
    Fills matched components with color 4.
    """
    def apply_logic(input_grid):
        input_grid = np.array(input_grid)
        h, w = input_grid.shape
        output_grid = input_grid.copy()
        
        visited = np.zeros((h, w), dtype=bool)
        
        for r in range(h):
            for c in range(w):
                if input_grid[r][c] == 0 and not visited[r][c]:
                    component = []
                    stack = [(r, c)]
                    visited[r][c] = True
                    
                    while stack:
                        curr_r, curr_c = stack.pop()
                        component.append((curr_r, curr_c))
                        
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if input_grid[nr][nc] == 0 and not visited[nr][nc]:
                                    visited[nr][nc] = True
                                    stack.append((nr, nc))
                    
                    # Check if this component is a rectangle
                    min_r = min(p[0] for p in component)
                    max_r = max(p[0] for p in component)
                    min_c = min(p[1] for p in component)
                    max_c = max(p[1] for p in component)
                    
                    expected_size = (max_r - min_r + 1) * (max_c - min_c + 1)
                    if len(component) == expected_size:
                        # Perfect Line Rule
                        is_perfect = True
                        
                        # Check Top and Bottom lines
                        for br in [min_r - 1, max_r + 1]:
                            if 0 <= br < h:
                                # All pixels from max(0, min_c-1) to min(w-1, max_c+1) must be 5
                                for bc in range(max(0, min_c - 1), min(w - 1, max_c + 1) + 1):
                                    if input_grid[br][bc] != 5:
                                        is_perfect = False; break
                                if not is_perfect: break
                                
                                # Pixels just outside must NOT be 5
                                for bc in [min_c - 2, max_c + 2]:
                                    if 0 <= bc < w:
                                        if input_grid[br][bc] == 5:
                                            is_perfect = False; break
                                if not is_perfect: break
                        
                        if not is_perfect: continue
                        
                        # Check Left and Right lines
                        for bc in [min_c - 1, max_c + 1]:
                            if 0 <= bc < w:
                                # All pixels from max(0, min_r-1) to min(h-1, max_r+1) must be 5
                                for br in range(max(0, min_r - 1), min(h - 1, max_r + 1) + 1):
                                    if input_grid[br][bc] != 5:
                                        is_perfect = False; break
                                if not is_perfect: break
                                
                                # Pixels just outside must NOT be 5
                                for br in [min_r - 2, max_r + 2]:
                                    if 0 <= br < h:
                                        if input_grid[br][bc] == 5:
                                            is_perfect = False; break
                                if not is_perfect: break
                        
                        if is_perfect:
                            for pr, pc in component:
                                output_grid[pr, pc] = 4
                                
        return output_grid

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
