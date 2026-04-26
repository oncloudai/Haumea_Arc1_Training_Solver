import numpy as np
from typing import List, Optional

def solve_grid_f8b3ba0a(input_grid):
    grid = np.array(input_grid)
    rows, cols = grid.shape
    color_counts = {}
    for r in range(rows):
        for c in range(cols - 1):
            if grid[r, c] != 0 and grid[r, c] == grid[r, c+1]:
                is_left_ok = (c == 0 or grid[r, c-1] == 0)
                if is_left_ok:
                    color = grid[r, c]
                    color_counts[color] = color_counts.get(color, 0) + 1
    if not color_counts: return np.zeros((0, 1), dtype=int)
    sorted_by_freq = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    others = sorted_by_freq[1:]
    output = np.array([[c] for c, count in others])
    return output

def solve_count_1x2_blocks_to_nx1(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_f8b3ba0a(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_f8b3ba0a(ti) for ti in solver.test_in]
    return None
