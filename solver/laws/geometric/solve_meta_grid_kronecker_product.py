import numpy as np
from typing import List, Optional

def solve_meta_grid_kronecker_product(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a grid of identical blocks (e.g., 3x3 blocks in a 9x9 bounding box).
    Derives a 3x3 meta-pattern P from which blocks are filled.
    The output is the Kronecker product of P with itself.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        # Find bounding box of all non-zero pixels
        rows, cols = np.where(grid != 0)
        if len(rows) == 0: return None
        r1, r2 = rows.min(), rows.max()
        c1, c2 = cols.min(), cols.max()
        
        h_bb = r2 - r1 + 1
        w_bb = c2 - c1 + 1
        
        # Check if bounding box is divisible by 3 (assuming 3x3 meta-grid)
        if h_bb % 3 != 0 or w_bb % 3 != 0: return None
        bh, bw = h_bb // 3, w_bb // 3
        
        sub = grid[r1:r2+1, c1:c2+1]
        P = np.zeros((3, 3), dtype=int)
        # Get the color too
        color = 0
        for i in range(3):
            for j in range(3):
                block = sub[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                if np.any(block != 0):
                    P[i, j] = 1
                    # Pick the first non-zero color
                    if color == 0:
                        color = block[block != 0][0]
                        
        if color == 0: return None
        
        # Kronecker product P x P
        out = np.zeros((3*3, 3*3), dtype=int)
        for i in range(3):
            for j in range(3):
                if P[i, j] == 1:
                    out[i*3:(i+1)*3, j*3:(j+1)*3] = P * color
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
