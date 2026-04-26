import numpy as np
from typing import List, Optional

def solve_rows_to_checkerboard(solver) -> Optional[List[np.ndarray]]:
    """
    Transforms two solid rows into an interlaced checkerboard pattern.
    Input consists of two rows of single colors.
    Output is a checkerboard of those two colors.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        if rows != 2: return None
        
        # Identify the two colors from the input rows
        color_a = grid[0, 0]
        color_b = grid[1, 0]
        
        # Verify rows are solid
        if not (np.all(grid[0, :] == color_a) and np.all(grid[1, :] == color_b)):
            return None
            
        row1 = [color_a if j % 2 == 0 else color_b for j in range(cols)]
        row2 = [color_b if j % 2 == 0 else color_a for j in range(cols)]
        return np.array([row1, row2])

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
