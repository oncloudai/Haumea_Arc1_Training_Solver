
import numpy as np
from typing import List, Optional

def solve_place_5_on_diagonal(solver) -> Optional[List[np.ndarray]]:
    """
    Task 6e02f1e3: Place 5s on a diagonal based on dominant value pattern.
    Rule 1: If grid is all one color, fill first row with 5.
    Rule 2: If counts are all equal, draw anti-diagonal of 5s.
    Rule 3: Otherwise, find dominant color and draw main diagonal of 5s.
    """
    def apply_logic(input_grid):
        input_grid = np.array(input_grid)
        output_grid = np.zeros_like(input_grid, dtype=int)
        unique, counts = np.unique(input_grid, return_counts=True)
        colors_counts = dict(zip(unique, counts))
        
        if len(colors_counts) == 1:
            output_grid[0, :] = 5
        elif len(set(counts)) == 1:
            np.fill_diagonal(np.fliplr(output_grid), 5)
        else:
            np.fill_diagonal(output_grid, 5)
        return output_grid

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        results.append(apply_logic(ti))
    return results
