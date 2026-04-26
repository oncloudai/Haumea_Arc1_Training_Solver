
import numpy as np
from typing import List, Optional

def solve_hollow_with_8_if_enclosed(solver) -> Optional[List[np.ndarray]]:
    """
    For each non-zero pixel, if all its 8-neighbors are also non-zero,
    change its color to 8.
    """
    def apply_logic(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        output = grid.copy()
        found_any = False
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if grid[r, c] != 0:
                    neighbors = grid[r-1:r+2, c-1:c+2]
                    if np.all(neighbors != 0):
                        if output[r, c] != 8:
                            output[r, c] = 8
                            found_any = True
        return output, found_any

    for inp, out in solver.pairs:
        pred, found = apply_logic(inp)
        if not found or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res, _ = apply_logic(ti)
        results.append(res)
    return results
