import numpy as np
from typing import List, Optional

def solve_replicate_row_sequence_downwards(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a color sequence from the first row of the grid.
    Fills all rows starting from a fixed offset (usually row 2) by cycling 
    through that sequence.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        if rows < 3: return None
        
        # Sequence from the first row
        sequence = grid[0].tolist()
        
        output = grid.copy()
        # Heuristic: Start filling from row 2
        for r in range(2, rows):
            idx = (r - 2) % len(sequence)
            color = sequence[idx]
            output[r, :] = color
        return output

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
