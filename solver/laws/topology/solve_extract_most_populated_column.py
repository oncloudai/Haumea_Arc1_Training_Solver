import numpy as np
from typing import List, Optional

def solve_extract_most_populated_column(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies the column with the maximum number of non-zero pixels.
    Returns a grid where only that column is preserved (or the column itself if it's a 1D task).
    In task d23f8c26: It preserves the column in a grid of the original size.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        
        # Count non-zero elements in each column
        non_zero_counts = [np.count_nonzero(grid[:, j]) for j in range(cols)]
        
        if not non_zero_counts or max(non_zero_counts) == 0: return None
        
        # Find the index of the maximum count (rightmost in case of tie)
        max_count = -1
        best_col = -1
        for j in range(cols):
            if non_zero_counts[j] >= max_count:
                max_count = non_zero_counts[j]
                best_col = j
        
        # Create output grid of the same size
        out = np.zeros_like(grid)
        if best_col != -1:
            out[:, best_col] = grid[:, best_col]
        return out

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
