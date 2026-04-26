
import numpy as np
from typing import List, Optional

def solve_replace_hollow_with_plus(solver) -> Optional[List[np.ndarray]]:
    """
    Finds all 3x3 hollow frames of color 1.
    Replaces each with a color 2 plus shape.
    """
    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        output = grid.copy()
        
        # Scan for 3x3 regions
        for r in range(rows - 2):
            for c in range(cols - 2):
                sub = grid[r:r+3, c:c+3]
                # A frame is 8 pixels of color 1 around a center that is NOT 1 (usually 0).
                is_frame = True
                for dr in range(3):
                    for dc in range(3):
                        if dr == 1 and dc == 1:
                            if sub[dr, dc] == 1: is_frame = False; break
                        else:
                            if sub[dr, dc] != 1: is_frame = False; break
                    if not is_frame: break
                
                if is_frame:
                    # Check if this frame is isolated (optional but ARC likes it)
                    # Let's just replace.
                    output[r:r+3, c:c+3] = 0
                    output[r, c+1] = 2
                    output[r+1, c] = 2
                    output[r+1, c+1] = 2
                    output[r+1, c+2] = 2
                    output[r+2, c+1] = 2
                    
        return output

    # Verify
    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    test_preds = []
    for inp in solver.test_in:
        pred = run_single(inp)
        test_preds.append(pred if pred is not None else np.array(inp))
        
    return test_preds
