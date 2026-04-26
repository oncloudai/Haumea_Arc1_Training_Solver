import numpy as np
from typing import List, Optional
from collections import Counter

def solve_grid_cells_to_output_flipped(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a grid structure in the input. Each cell in the grid maps to a pixel
    in the output. The output grid is horizontally flipped relative to the input cells.
    If input cell (r, c) contains color X, output[r, num_cols-1-c] = X.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # 1. Find divider color and boundaries
        # A divider is a color that appears in many rows or many columns as a line.
        best_color = -1
        best_row_divs = []
        best_col_divs = []
        
        for color in range(1, 10):
            row_divs = [r for r in range(h) if np.all(grid[r, :] == color)]
            col_divs = [c for c in range(w) if np.all(grid[:, c] == color)]
            
            if len(row_divs) >= 1 and len(col_divs) >= 1:
                if len(row_divs) + len(col_divs) > len(best_row_divs) + len(best_col_divs):
                    best_color = color
                    best_row_divs = row_divs
                    best_col_divs = col_divs
                    
        if best_color == -1:
            return None
            
        # Get intervals
        row_starts = [0] + [r + 1 for r in best_row_divs]
        row_ends = best_row_divs + [h]
        row_intervals = [(s, e) for s, e in zip(row_starts, row_ends) if e > s]
        
        col_starts = [0] + [c + 1 for c in best_col_divs]
        col_ends = best_col_divs + [w]
        col_intervals = [(s, e) for s, e in zip(col_starts, col_ends) if e > s]
        
        num_rows = len(row_intervals)
        num_cols = len(col_intervals)
        
        out = np.zeros((num_rows, num_cols), dtype=int)
        
        for r_idx, (r_start, r_end) in enumerate(row_intervals):
            for c_idx, (c_start, c_end) in enumerate(col_intervals):
                sub = grid[r_start:r_end, c_start:c_end]
                # Find most frequent color in sub that is not 0 or divider
                unique, counts = np.unique(sub, return_counts=True)
                colors = {u: count for u, count in zip(unique, counts) if u != 0 and u != best_color}
                if colors:
                    dominant_color = max(colors, key=colors.get)
                    # Horizontal flip
                    out[r_idx, num_cols - 1 - c_idx] = dominant_color
                    
        return out

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
