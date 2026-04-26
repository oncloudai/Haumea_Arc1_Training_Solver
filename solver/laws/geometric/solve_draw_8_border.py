
import numpy as np
from typing import List, Optional

def solve_draw_8_border(solver) -> Optional[List[np.ndarray]]:
    """
    Draw an 8-border around the grid edges, leaving interior as 0.
    """
    def apply_logic(input_grid):
        input_grid = np.array(input_grid)
        rows, cols = input_grid.shape
        output_grid = np.zeros_like(input_grid, dtype=int)
        output_grid[0, :] = 8
        output_grid[rows-1, :] = 8
        output_grid[:, 0] = 8
        output_grid[:, cols-1] = 8
        return output_grid

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        results.append(apply_logic(ti))
    return results
