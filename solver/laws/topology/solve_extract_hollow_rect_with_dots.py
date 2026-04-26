import numpy as np
from typing import List, Optional

def solve_grid_fcb5c309(input_grid):
    grid = np.array(input_grid)
    rows, cols = grid.shape
    
    unique_colors = np.unique(grid)
    colors = [c for c in unique_colors if c != 0]
    
    def is_hollow_rect(c, r1, r2, c1, c2):
        # Check boundary
        for r in range(r1, r2 + 1):
            if grid[r, c1] != c or grid[r, c2] != c:
                return False
        for c_ in range(c1, c2 + 1):
            if grid[r1, c_] != c or grid[r2, c_] != c:
                return False
        return True

    best_rect = None
    max_area = -1
    
    for c in colors:
        # Find all possible rectangles of color c
        # We can just iterate over all pairs of points of color c
        coords = np.argwhere(grid == c)
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                r1, c1 = coords[i]
                r2, c2 = coords[j]
                
                if r1 > r2: r1, r2 = r2, r1
                if c1 > c2: c1, c2 = c2, c1
                
                if r2 - r1 < 2 or c2 - c1 < 2: continue # Too small to be a hollow rectangle
                
                area = (r2 - r1 + 1) * (c2 - c1 + 1)
                if area <= max_area: continue
                
                if is_hollow_rect(c, r1, r2, c1, c2):
                    max_area = area
                    best_rect = (c, r1, r2, c1, c2)

    if best_rect is None:
        return np.zeros((0,0), dtype=int)
        
    f_color, r1, r2, c1, c2 = best_rect
    h, w = r2 - r1 + 1, c2 - c1 + 1
    
    # Identify the dot color
    dot_color = None
    for c in colors:
        if c != f_color:
            dot_color = c
            break
    if dot_color is None:
        # If no other color, we might be in trouble, but let's assume it's f_color?
        # Or maybe it's one of the other examples.
        dot_color = f_color

    output = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            # If on boundary
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                output[r, c] = dot_color
            # If input has a dot color at this relative position
            if grid[r + r1, c + c1] == dot_color:
                output[r, c] = dot_color
                
    return output

def solve_extract_hollow_rect_with_dots(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_fcb5c309(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_fcb5c309(ti) for ti in solver.test_in]
    return None
