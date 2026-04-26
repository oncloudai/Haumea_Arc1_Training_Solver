import numpy as np
from typing import List, Optional

def solve_concentric_repeating_shells(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies concentric shells in the input, extracts their colors,
    and repeats them in a larger output grid.
    Used for task 539a4f51.
    """
    def run_single(grid, out_shape=None):
        grid = np.array(grid)
        rows, cols = grid.shape
        if rows != cols: return None
        N = rows
        
        # Extract shell colors from diagonal
        # (Assuming shells are consistent, which they are in this task)
        S = []
        for i in range(N):
            c = grid[i, i]
            if c != 0:
                S.append(c)
        
        if not S: return None
        
        if out_shape is None:
            # For test case, assume 2*N
            out_rows, out_cols = 2 * rows, 2 * cols
        else:
            out_rows, out_cols = out_shape
            
        out = np.zeros((out_rows, out_cols), dtype=int)
        L = len(S)
        for r in range(out_rows):
            for c in range(out_cols):
                dist = max(r, c)
                out[r, c] = S[dist % L]
        return out

    results = []
    for inp, out_expected in solver.pairs:
        pred = run_single(inp, out_expected.shape)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    for ti in solver.test_in:
        # We need to guess the output shape for the test input.
        # In this task, it's always 2*N.
        guessed_shape = (2 * ti.shape[0], 2 * ti.shape[1])
        results.append(run_single(ti, guessed_shape))
    return results
