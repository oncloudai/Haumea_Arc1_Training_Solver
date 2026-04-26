import numpy as np
from typing import List, Optional

def solve_stack_row_segments_by_length_right_aligned(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies non-zero segments in each row.
    Counts the length and identifies the color of each row's segment.
    Sorts segments by length ascending.
    Re-stacks segments into an output grid of the same size, starting from the bottom row,
    aligning each segment to the right edge.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        
        # Identify strips
        strips = []
        for r in range(rows):
            row = grid[r]
            non_zeros = row[row != 0]
            if len(non_zeros) > 0:
                # Assume one color per row based on patterns
                color = non_zeros[0]
                length = len(non_zeros)
                strips.append((color, length))
                
        # Sort strips by length ascending
        strips.sort(key=lambda x: x[1])
        
        # Create output grid filled with 0
        output = np.zeros((rows, cols), dtype=int)
        
        # Place strips from the bottom up, reversed (longest at the bottom)
        for i, (color, length) in enumerate(reversed(strips)):
            r = rows - 1 - i
            if r >= 0:
                # Right align: starts at cols - length
                output[r, max(0, cols-length):] = color
                
        return output

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
