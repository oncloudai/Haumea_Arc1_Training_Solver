
import numpy as np
from typing import List, Optional
from collections import Counter

def solve_fill_lines_between_opposite_edges(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies the color that most frequently has matching pixels on opposite edges of the grid.
    Fills the entire row or column connecting these pixels for that specific color ONLY.
    The output contains ONLY these lines (and background 0).
    """
    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        
        pair_colors = []
        # Check rows: match at col 0 and last col
        for r in range(rows):
            if grid[r, 0] != 0 and grid[r, 0] == grid[r, cols - 1]:
                pair_colors.append(grid[r, 0])
        # Check columns: match at row 0 and last row
        for c in range(cols):
            if grid[0, c] != 0 and grid[0, c] == grid[rows - 1, c]:
                pair_colors.append(grid[0, c])
                
        if not pair_colors: return None
        
        # Consistent rule for this task: only use the color that forms the most edge-pairs.
        most_common_color = Counter(pair_colors).most_common(1)[0][0]
        
        output = np.zeros_like(grid)
        found_any = False
        
        for r in range(rows):
            if grid[r, 0] == most_common_color and grid[r, 0] == grid[r, cols - 1]:
                output[r, :] = most_common_color
                found_any = True
        for c in range(cols):
            if grid[0, c] == most_common_color and grid[0, c] == grid[rows - 1, c]:
                output[:, c] = most_common_color
                found_any = True
                
        return output if found_any else None

    # Verify on pairs
    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    test_preds = []
    for inp in solver.test_in:
        pred = run_single(inp)
        test_preds.append(pred if pred is not None else np.zeros_like(inp))
        
    return test_preds
