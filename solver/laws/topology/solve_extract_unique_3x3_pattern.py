import numpy as np
from typing import List, Optional

def solve_grid_a87f7484(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    
    colors = np.unique(grid)
    patterns = {} # pattern_tuple -> color
    pattern_counts = {} # pattern_tuple -> count
    color_blocks = {} # color -> 3x3 grid
    
    for c in colors:
        if c == 0:
            continue
        
        # Find pixels of color c
        rows, cols = np.where(grid == c)
        if len(rows) == 0:
            continue
            
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()
        
        # Extract 3x3 block
        block = np.zeros((3, 3), dtype=int)
        for r, cc in zip(rows, cols):
            if 0 <= r - r_min < 3 and 0 <= cc - c_min < 3:
                block[r - r_min, cc - c_min] = c
            
        # Pattern (binary)
        pattern = tuple(tuple(int(val != 0) for val in row) for row in block)
        
        patterns[pattern] = c
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        color_blocks[c] = block
        
    # Find unique pattern
    winner_color = -1
    for pattern, count in pattern_counts.items():
        if count == 1:
            winner_color = patterns[pattern]
            break
            
    if winner_color != -1:
        return color_blocks[winner_color]
    else:
        return np.zeros((0, 0), dtype=int)

def solve_extract_unique_3x3_pattern(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_a87f7484(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_a87f7484(ti) for ti in solver.test_in]
    return None
