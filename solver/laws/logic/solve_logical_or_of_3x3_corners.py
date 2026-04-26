import numpy as np
from typing import List, Optional

def solve_logical_or_of_3x3_corners(solver) -> Optional[List[np.ndarray]]:
    """
    Extracts the four 3x3 corners of the input grid.
    Performs an element-wise logical OR (maximum) of all four corners.
    Returns the resulting 3x3 grid.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        if rows < 3 or cols < 3: return None
        
        tl = grid[0:3, 0:3]
        tr = grid[0:3, cols-3:cols]
        bl = grid[rows-3:rows, 0:3]
        br = grid[rows-3:rows, cols-3:cols]
        
        # Element-wise maximum (Logical OR for ARC colors)
        output = np.maximum(np.maximum(tl, tr), np.maximum(bl, br))
        return output

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out_expected.shape or not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
