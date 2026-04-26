
import numpy as np
from typing import List, Optional

def solve_extract_top_right_3x3(solver) -> Optional[List[np.ndarray]]:
    """
    Select the top-right 3x3 block of a 9x9 grid.
    """
    def apply_logic(input_grid):
        input_grid = np.array(input_grid)
        if input_grid.shape != (9, 9): return None
        return input_grid[0:3, 6:9]

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
