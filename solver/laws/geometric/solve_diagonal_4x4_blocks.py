
import numpy as np
from typing import List, Optional

def solve_diagonal_4x4_blocks(solver) -> Optional[List[np.ndarray]]:
    """
    Find a 2x2 block in the 3x3 input that contains color 2 and another color (usually 3).
    Scale this 2x2 such that each pixel becomes a 4x4 block in a 9x9 output.
    Keep only the two 4x4 blocks belonging to the diagonal that contained the color 2.
    """
    if not all(inp.shape == (3, 3) for inp, out in solver.pairs): return None
    if not all(out.shape == (9, 9) for inp, out in solver.pairs): return None
    
    def apply_logic(grid):
        grid = np.array(grid)
        best_sub = None
        best_r, best_c = -1, -1
        max_non_zero = -1
        
        for r in range(2):
            for c in range(2):
                sub = grid[r:r+2, c:c+2]
                if 2 in sub:
                    non_zero = np.count_nonzero(sub)
                    if non_zero > max_non_zero:
                        # Check if it has another color besides 0 and 2
                        others = np.unique(sub)
                        others = others[(others != 0) & (others != 2)]
                        if len(others) > 0:
                            max_non_zero = non_zero
                            best_sub = sub
                            best_r, best_c = r, c
        
        if best_sub is not None:
            others = np.unique(best_sub)
            fill_color = others[(others != 0) & (others != 2)][0]
            res = np.zeros((9, 9), dtype=int)
            r, c = best_r, best_c
            if best_sub[0,0] == 2 or best_sub[1,1] == 2:
                res[r:r+4, c:c+4] = fill_color
                res[r+4:r+8, c+4:c+8] = fill_color
            elif best_sub[0,1] == 2 or best_sub[1,0] == 2:
                res[r:r+4, c+4:c+8] = fill_color
                res[r+4:r+8, c:c+4] = fill_color
            return res
        return None

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
