import numpy as np
from typing import List, Optional

def solve_repeat_3x3_stamps_based_on_pixel_count(solver) -> Optional[List[np.ndarray]]:
    """
    Given a 3x3 input, the output size is determined by the number of non-zero pixels P.
    N = 9 - P. Output size is (3*N) x (3*N).
    P stamps of the 3x3 input are placed in the output NxN meta-grid, filling row by row.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        if grid.shape != (3, 3): return None
        
        non_zero_pixels = np.where(grid != 0)
        P = len(non_zero_pixels[0])
        if P == 0 or P >= 9: return None
        
        N = 9 - P
        out_size = 3 * N
        out = np.zeros((out_size, out_size), dtype=int)
        
        # Place P stamps in the NxN grid
        for k in range(P):
            r_meta = k // N
            c_meta = k % N
            if r_meta >= N: break # Should not happen based on N = 9-P and P <= 4 (wait, P can be larger)
            # Actually, P could be larger than N if P > N.
            # 6 -> 3. 6 stamps, 3x3 grid. OK.
            # 5 -> 4. 5 stamps, 4x4 grid. OK.
            # 4 -> 5. 4 stamps, 5x5 grid. OK.
            # 3 -> 6. 3 stamps, 6x6 grid. OK.
            # 2 -> 7. 2 stamps, 7x7 grid. OK.
            # In all these cases, P < N^2.
            
            out[r_meta*3 : (r_meta+1)*3, c_meta*3 : (c_meta+1)*3] = grid
            
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
