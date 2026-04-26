import numpy as np
from typing import List, Optional

def solve_subgrid_kronecker_self_product(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies the smallest subgrid P containing all non-zero pixels in the input.
    The output is the Kronecker product of P with itself.
    If P is (ph x pw), the output is (ph*ph x pw*pw).
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = np.where(grid != 0)
        if len(rows) == 0: return None
        
        r1, r2 = rows.min(), rows.max()
        c1, c2 = cols.min(), cols.max()
        
        P = grid[r1:r2+1, c1:c2+1]
        ph, pw = P.shape
        
        # Check if Kronecker product size matches input or a known standard
        # In this task, output is always the same size as the Kronecker product
        out_h, out_w = ph * ph, pw * pw
        
        out = np.zeros((out_h, out_w), dtype=int)
        for r in range(ph):
            for c in range(pw):
                if P[r, c] != 0:
                    out[r*ph : (r+1)*ph, c*pw : (c+1)*pw] = P
        return out

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
