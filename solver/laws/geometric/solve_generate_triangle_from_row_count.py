import numpy as np
from typing import List, Optional

def solve_generate_triangle_from_row_count(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a non-zero color and its initial count in the first row.
    Generates a triangle of rows where the count increases by 1 each row.
    Each row fills the first 'count' columns with that color.
    The number of output rows is half the column width.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        if grid.ndim < 2: return None
        in_row = grid[0]
        cols = len(in_row)
        
        # Non-zero color and start count
        non_zeros = in_row[in_row != 0]
        if len(non_zeros) == 0: return None
        color = int(non_zeros[0])
        start_count = int(np.sum(in_row != 0))
            
        out_rows = cols // 2
        output = np.zeros((out_rows, cols), dtype=int)
        for i in range(out_rows):
            count = start_count + i
            output[i, :min(count, cols)] = color
        return output

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out_expected.shape or not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
