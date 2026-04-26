import numpy as np
from typing import List, Optional

def solve_count_magenta_3x3_to_blue(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies 3x3 regions separated by dividers (at index 3 and 7).
    Counts magenta (6) pixels in each 3x3 region.
    If the count is exactly 2, the corresponding output cell is blue (1), else zero.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        if rows != 11 or cols != 11: return None
        
        output = np.zeros((3, 3), dtype=int)
        # Slices for the nine 3x3 subgrids
        row_slices = [(0, 3), (4, 7), (8, 11)]
        col_slices = [(0, 3), (4, 7), (8, 11)]
        
        for i, (rs, re) in enumerate(row_slices):
            for j, (cs, ce) in enumerate(col_slices):
                sub = grid[rs:re, cs:ce]
                if np.sum(sub == 6) == 2:
                    output[i, j] = 1
                else:
                    output[i, j] = 0
        return output

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
