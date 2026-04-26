
import numpy as np
from typing import List, Optional

def solve_fill_modal_value(solver) -> Optional[List[np.ndarray]]:
    """
    Find the most frequent non-zero value in the input grid;
    fill the entire output grid with that value.
    """
    def apply_logic(input_grid):
        grid = np.array(input_grid)
        unique, counts = np.unique(grid, return_counts=True)
        # Filter out 0 (background)
        non_zero = [(u, c) for u, c in zip(unique, counts) if u != 0]
        if not non_zero: return grid
        # Find the one with max count
        modal_value = max(non_zero, key=lambda x: x[1])[0]
        return np.full_like(grid, modal_value)

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        results.append(apply_logic(ti))
    return results
