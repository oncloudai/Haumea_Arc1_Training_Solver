
import numpy as np
from typing import List, Optional

def solve_fill_3x3_enclosures_with_blue(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies all 3x3 areas of 0s. 
    Fills them with color 1.
    If multiple 3x3s overlap, filling the first one (in scan order) 
    makes the subsequent ones no longer qualify as all-zero.
    """
    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        output = grid.copy()
        found_any = False
        
        # Scan for 3x3 empty regions
        for r in range(rows - 2):
            for c in range(cols - 2):
                sub = output[r:r+3, c:c+3]
                if np.all(sub == 0):
                    output[r:r+3, c:c+3] = 1
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
        test_preds.append(pred if pred is not None else np.array(inp))
        
    return test_preds
