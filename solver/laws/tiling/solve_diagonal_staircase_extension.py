
import numpy as np
from typing import List, Optional

def solve_diagonal_staircase_extension(solver) -> Optional[List[np.ndarray]]:
    """
    Extend the grid by assuming Row[i] = Row[i-P] shifted by S.
    """
    def get_p_s(grid):
        h, w = grid.shape
        for P in range(1, h):
            for S in range(-w + 1, w):
                consistent = True
                for i in range(P, h):
                    expected = np.zeros(w, dtype=int)
                    if S >= 0:
                        expected[S:] = grid[i-P, :w-S]
                    else:
                        expected[:w+S] = grid[i-P, -S:]
                    if not np.array_equal(grid[i], expected):
                        consistent = False; break
                if consistent: return P, S
        return None, None

    def apply_logic(inp, out_h, out_w, P, S):
        h, w = inp.shape
        out = np.zeros((out_h, out_w), dtype=int)
        out[:h, :w] = inp
        for i in range(h, out_h):
            row_prev = out[i-P]
            new_row = np.zeros(out_w, dtype=int)
            if S >= 0:
                new_row[S:] = row_prev[:out_w-S]
            else:
                new_row[:out_w+S] = row_prev[-S:]
            out[i] = new_row
        return out

    # Check consistency across training pairs
    results = []
    target_h = solver.train_out[0].shape[0]
    
    for inp, out in solver.pairs:
        P, S = get_p_s(inp)
        if P is None: return None
        pred = apply_logic(inp, out.shape[0], out.shape[1], P, S)
        if not np.array_equal(pred, out): return None
        
    for ti in solver.test_in:
        P, S = get_p_s(ti)
        if P is None: return None
        res = apply_logic(ti, target_h, ti.shape[1], P, S)
        results.append(res)
    return results
