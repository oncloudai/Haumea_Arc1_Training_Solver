
import numpy as np
from typing import List, Optional

def solve_multicolor_pattern_masked_by_grid(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a multicolor square pattern P of size S x S (S=3 or 4).
    P is selected as the square with the most colors and fewest zeros.
    Identifies a grid of blocks of a single color C_grid.
    The abstract grid is also S x S.
    The output is the pattern P masked by the presence of blocks in the grid.
    """
    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        
        # 1. Find the multicolor pattern P (3x3 or 4x4)
        best_p = None
        best_s = -1
        best_loc = (-1, -1)
        max_colors = -1
        min_zeros = 1000
        
        for s in [3, 4]:
            for r in range(rows - s + 1):
                for c in range(cols - s + 1):
                    sub = grid[r:r+s, c:c+s]
                    unique = np.unique(sub)
                    num_colors = len(unique[unique != 0])
                    num_zeros = np.sum(sub == 0)
                    
                    if num_colors >= 3:
                        # Criteria: More colors first, then fewer zeros, then bottom-right
                        if (num_colors > max_colors) or \
                           (num_colors == max_colors and num_zeros < min_zeros) or \
                           (num_colors == max_colors and num_zeros == min_zeros and r + c > best_loc[0] + best_loc[1]):
                            max_colors = num_colors
                            min_zeros = num_zeros
                            best_p = sub.copy()
                            best_s = s
                            best_loc = (r, c)
                            
        if best_p is None: return None
        s = best_s
        pr, pc = best_loc
        
        # 2. Identify C_grid
        grid_copy = grid.copy()
        grid_copy[pr:pr+s, pc:pc+s] = 0
        unique_colors, counts = np.unique(grid_copy[grid_copy != 0], return_counts=True)
        if len(unique_colors) == 0: return None
        c_grid = unique_colors[np.argmax(counts)]
        
        # 3. Find BBox of C_grid
        coords = np.argwhere(grid_copy == c_grid)
        if len(coords) == 0: return None
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        gh, gw = r_max - r_min + 1, c_max - c_min + 1
        
        if gh % s != 0 or gw % s != 0:
            return None
            
        bh, bw = gh // s, gw // s
        
        # 4. Determine Mask
        output = np.zeros((s, s), dtype=int)
        for R in range(s):
            for C in range(s):
                sub_block = grid_copy[r_min + R*bh : r_min + (R+1)*bh, c_min + C*bw : c_min + (C+1)*bw]
                if np.any(sub_block == c_grid):
                    output[R, C] = best_p[R, C]
                    
        return output

    # Verify on pairs
    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    test_preds = []
    for inp in solver.test_in:
        pred = run_single(inp)
        test_preds.append(pred if pred is not None else np.zeros((1,1), dtype=int))
        
    return test_preds
