
import numpy as np
from collections import defaultdict
from typing import List, Optional

def solve_subgrid_unique_marker_presence(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies 3x3 blocks.
    A block at (R, C) contributes a 1 to the output if it has a marker (6)
    at a relative position (dr, dc) that NO OTHER block has.
    """
    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        if rows != 11 or cols != 11: return None
        
        # Fixed grid for this task
        row_starts = [0, 4, 8]
        col_starts = [0, 4, 8]
        
        by_rel = defaultdict(list)
        for R in range(3):
            for C in range(3):
                rs, cs = row_starts[R], col_starts[C]
                for dr in range(3):
                    for dc in range(3):
                        if grid[rs + dr, cs + dc] == 6:
                            by_rel[(dr, dc)].append((R, C))
                            
        output = np.zeros((3, 3), dtype=int)
        for rel, blocks in by_rel.items():
            if len(blocks) == 1:
                R, C = blocks[0]
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
