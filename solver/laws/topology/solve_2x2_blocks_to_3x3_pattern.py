import numpy as np
from typing import List, Optional

def solve_grid_ff28f65a(input_grid):
    grid = np.array(input_grid)
    rows, cols = grid.shape
    blocks = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            if np.all(grid[r:r+2, c:c+2] == 2): blocks.append((r, c))
    blocks.sort()
    sequence = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
    output = np.zeros((3, 3), dtype=int)
    for i in range(min(len(blocks), len(sequence))):
        r, c = sequence[i]; output[r, c] = 1
    return output

def solve_2x2_blocks_to_3x3_pattern(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_ff28f65a(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_ff28f65a(ti) for ti in solver.test_in]
    return None
