
import numpy as np
from typing import List, Optional

def solve_extract_unique_asymmetric_3x3_subgrid(solver) -> Optional[List[np.ndarray]]:
    """
    Divides the input grid into 3x3 blocks.
    Identifies the unique block that is NOT symmetric across its main diagonal (i.e., not equal to its transpose).
    Outputs that 3x3 block.
    """
    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        if rows % 3 != 0 or cols % 3 != 0: return None
        
        blocks = []
        for r in range(0, rows, 3):
            for c in range(0, cols, 3):
                blocks.append(grid[r:r+3, c:c+3])
        
        if len(blocks) < 2: return None
        
        is_symmetric = [np.array_equal(b, b.T) for b in blocks]
        
        # Find index of the unique asymmetric block
        candidates = [i for i, sym in enumerate(is_symmetric) if not sym]
        
        if len(candidates) == 1:
            return blocks[candidates[0]]
        
        # If no asymmetric blocks, maybe the unique symmetric one?
        # In this task, it's always unique asymmetric.
        return None

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
