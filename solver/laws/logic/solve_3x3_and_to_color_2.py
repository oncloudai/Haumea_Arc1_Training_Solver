import numpy as np
from typing import List, Optional

def solve_grid_fafffa47(input_grid):
    grid = np.array(input_grid)
    if grid.shape != (6, 3): return grid
    grid1 = grid[0:3, 0:3]
    grid2 = grid[3:6, 0:3]
    output = np.zeros((3, 3), dtype=int)
    for r in range(3):
        for c in range(3):
            if grid1[r, c] == 0 and grid2[r, c] == 0: output[r, c] = 2
    return output

def solve_3x3_and_to_color_2(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        if inp.shape != (6, 3) or out.shape != (3, 3): consistent = False; break
        res = solve_grid_fafffa47(inp)
        if not np.array_equal(res, out): consistent = False; break
            
    if consistent:
        return [solve_grid_fafffa47(ti) for ti in solver.test_in]
    return None
