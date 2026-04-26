import numpy as np
from typing import List, Optional

def solve_diagonal_periodicity_completion(solver) -> Optional[List[np.ndarray]]:
    """
    Infers a fundamental diagonal pattern (c + r) % p_len from non-zero pixels.
    Completes the grid by propagating this inferred pattern to all cells.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        
        best_p_len = -1
        best_pattern = []
        
        # Try all possible pattern lengths from 1 to cols
        for p_len in range(1, cols + 1):
            pattern = [0] * p_len
            consistent = True
            for r in range(rows):
                for c in range(cols):
                    if grid[r, c] != 0:
                        idx = (c + r) % p_len
                        color = int(grid[r, c])
                        if pattern[idx] != 0 and pattern[idx] != color:
                            consistent = False
                            break
                        pattern[idx] = color
                if not consistent: break
            
            # Check if the pattern is complete (no zeros)
            if consistent and 0 not in pattern:
                best_p_len = p_len
                best_pattern = pattern
                break
                
        if best_p_len == -1: return None
            
        output = np.zeros((rows, cols), dtype=int)
        for r in range(rows):
            for c in range(cols):
                output[r, c] = best_pattern[(c + r) % best_p_len]
        return output

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
