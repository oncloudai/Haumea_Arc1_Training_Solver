
import numpy as np
from collections import defaultdict
from typing import List, Optional

def solve_subgrid_unique_pixel_presence(solver) -> Optional[List[np.ndarray]]:
    """
    Divides the grid into blocks based on a grid-like divider (color 8).
    Identifies 'marker' pixels (color 6).
    A block at (R, C) in the block layout results in a 1 in the output at (R, C)
    if it contains a marker pixel at a relative position (dr, dc) that is NOT
    occupied by a marker pixel in any other block.
    """
    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        
        # 1. Find the divider color and grid layout
        # The divider is a color that forms at least two full rows or columns
        # In this task it's color 8.
        divider_color = 8
        if not np.any(grid == divider_color): return None
        
        div_rows = []
        for r in range(rows):
            if np.all(grid[r, :] == divider_color): div_rows.append(r)
        div_cols = []
        for c in range(cols):
            if np.all(grid[:, c] == divider_color): div_cols.append(c)
            
        if not div_rows or not div_cols: return None
        
        # Block boundaries
        row_bounds = [0] + [r+1 for r in div_rows] + [rows]
        col_bounds = [0] + [c+1 for c in div_cols] + [cols]
        
        # Extract blocks and their relative markers
        blocks = []
        num_R = len(row_bounds) - 1
        num_C = len(col_bounds) - 1
        
        # In this task, the output is always 3x3.
        # This implies there are exactly 9 blocks.
        if num_R != 3 or num_C != 3: return None
        
        # We need a fixed marker color. In this task it's 6.
        marker_color = 6
        
        by_rel = defaultdict(list)
        for R in range(num_R):
            for C in range(num_C):
                block = grid[row_bounds[R]:row_bounds[R+1]-1, col_bounds[C]:col_bounds[C+1]-1]
                # Note: -1 because div_rows/cols are included in bounds
                # Actually, the logic is:
                # bounds = [0, 3, 7, 11] for rows 3, 7 being dividers.
                # rows 0,1,2 | divider 3 | rows 4,5,6 | divider 7 | rows 8,9,10
                
                # Let's fix the bounds
                r_start = row_bounds[R]
                r_end = row_bounds[R+1] - (1 if R < num_R - 1 else 0)
                # Wait, if row_bounds = [0, 4, 8, 11]
                # R=0: 0 to 3. R=1: 4 to 7. R=2: 8 to 10.
                # Yes.
                
                # Let's just find the relative positions of 6 in each block
                for r in range(row_bounds[R], row_bounds[R+1]-1 if R < num_R-1 else rows):
                    for c in range(col_bounds[C], col_bounds[C+1]-1 if C < num_C-1 else cols):
                        if grid[r, c] == marker_color:
                            dr = r - row_bounds[R]
                            dc = c - col_bounds[C]
                            by_rel[(dr, dc)].append((R, C))
                            
        output = np.zeros((num_R, num_C), dtype=int)
        for rel, block_indices in by_rel.items():
            if len(block_indices) == 1:
                R, C = block_indices[0]
                output[R, C] = 1
                
        return output

    # Verify on pairs
    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    test_preds = []
    for inp in solver.test_in:
        pred = run_single(inp)
        test_preds.append(pred if pred is not None else np.zeros((3,3), dtype=int))
        
    return test_preds
