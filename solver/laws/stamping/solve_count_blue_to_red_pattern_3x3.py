import numpy as np
from typing import List, Optional

def solve_grid_794b24be(grid):
    grid = np.array(grid)
    count = np.sum(grid == 1)
    out = np.zeros((3, 3), dtype=int)
    order = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
    for i in range(min(count, len(order))):
        r, c = order[i]
        out[r, c] = 2
    return out

def solve_count_blue_to_red_pattern_3x3(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_794b24be(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_794b24be(ti) for ti in solver.test_in]
    return None
