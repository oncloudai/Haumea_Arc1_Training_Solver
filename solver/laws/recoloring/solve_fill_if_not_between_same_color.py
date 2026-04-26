import numpy as np
from typing import List, Optional

def solve_fill_if_not_between_same_color(solver) -> Optional[List[np.ndarray]]:
    """
    Fills 0s with color 1 if they are NOT 'enclosed' by two pixels of the same color
    in the same row or column, considering wrapping.
    Specifically, for each 0 at (r, c):
    - If there exist c1, c2 such that (r, c) is between (r, c1) and (r, c2) 
      and grid[r, c1] == grid[r, c2] != 0, it stays 0.
    - If there exist r1, r2 such that (r, c) is between (r1, c) and (r2, c) 
      and grid[r1, c] == grid[r2, c] != 0, it stays 0.
    - Otherwise, it becomes color 1.
    """
    def is_between(x, x1, x2, length):
        if x1 < x2:
            return x1 < x < x2
        else: # Wrapping
            return x > x1 or x < x2

    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        out = grid.copy()
        
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 0:
                    enclosed = False
                    
                    # Check row
                    row_pixels = [(cc, grid[r, cc]) for cc in range(w) if grid[r, cc] != 0]
                    for i in range(len(row_pixels)):
                        for j in range(len(row_pixels)):
                            if i == j: continue
                            c1, color1 = row_pixels[i]
                            c2, color2 = row_pixels[j]
                            if color1 == color2:
                                if is_between(c, c1, c2, w):
                                    enclosed = True
                                    break
                        if enclosed: break
                    
                    if not enclosed:
                        # Check col
                        col_pixels = [(rr, grid[rr, c]) for rr in range(h) if grid[rr, c] != 0]
                        for i in range(len(col_pixels)):
                            for j in range(len(col_pixels)):
                                if i == j: continue
                                r1, color1 = col_pixels[i]
                                r2, color2 = col_pixels[j]
                                if color1 == color2:
                                    if is_between(r, r1, r2, h):
                                        enclosed = True
                                        break
                            if enclosed: break
                            
                    if not enclosed:
                        out[r, c] = 1
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
