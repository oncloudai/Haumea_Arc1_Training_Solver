import numpy as np
from typing import List, Optional
from collections import Counter

def solve_grid_a78176bb(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    
    # Find main color
    colors = np.unique(grid)
    main_colors = [c for c in colors if c != 0 and c != 5]
    if not main_colors:
        return grid
    main_color = main_colors[0]
    
    # Find diagonal offset
    # c = (r + d) % w
    pixels = np.argwhere(grid == main_color)
    offsets = []
    for r, c in pixels:
        offsets.append((c - r) % w)
    
    if not offsets: return grid
    d = Counter(offsets).most_common(1)[0][0]
    
    output_grid = np.zeros_like(grid)
    d2 = (d + 4) % w
    
    for r in range(h):
        c1 = (r + d) % w
        c2 = (r + d2) % w
        output_grid[r, c1] = main_color
        output_grid[r, c2] = main_color
        
    return output_grid

def solve_draw_diagonal_lines_by_offset(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_a78176bb(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_a78176bb(ti) for ti in solver.test_in]
    return None
