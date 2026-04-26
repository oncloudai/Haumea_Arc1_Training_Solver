import numpy as np
from typing import List, Optional

def solve_grid_feca6190(input_grid):
    grid = np.array(input_grid)
    v = grid[0]
    m = len(v)
    
    num_non_zero = np.sum(v != 0)
    n = m * num_non_zero
    
    output = np.zeros((n, n), dtype=int)
    for r in range(n):
        # The offset is n - 1 - r? Let's check row n-1.
        # Row n-1 starts at col 0. (n-1-r) = 0.
        # Row 0 starts at col n-1. (n-1-r) = n-1.
        start_col = n - 1 - r
        for i in range(m):
            if 0 <= start_col + i < n:
                output[r, start_col + i] = v[i]
                
    return output

def solve_skew_vector_to_diagonal_matrix(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_feca6190(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_feca6190(ti) for ti in solver.test_in]
    return None
