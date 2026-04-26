import numpy as np
from typing import List, Optional

def solve_scaled_object_with_rays(solver) -> Optional[List[np.ndarray]]:
    """
    1. Identify border (last row and column).
    2. F = number of unique colors in border + 1.
    3. Upscale grid by factor F.
    4. In the upscaled non-border area, draw Red (2) rays along specific diagonals.
    Lines: r-c = r1-c1, r+c = r1+c2 (where r1,c1,r2,c2 are object bounds).
    Rays are only drawn for pixels outside the object rectangle.
    """
    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        if rows < 3 or cols < 3: return None
        
        # Border colors (excluding 0)
        border_colors = set()
        for r in range(rows): border_colors.add(grid[r, cols-1])
        for c in range(cols): border_colors.add(grid[rows-1, c])
        if 0 in border_colors: border_colors.remove(0)
        
        F = len(border_colors) + 1
        if F < 2: return None
        
        # Upscale
        out_rows, out_cols = rows * F, cols * F
        output = np.zeros((out_rows, out_cols), dtype=int)
        for r in range(rows):
            for c in range(cols):
                output[r*F:(r+1)*F, c*F:(c+1)*F] = grid[r, c]
                
        # Non-border area in output
        nb_rows, nb_cols = (rows-1) * F, (cols-1) * F
        
        # Identify object in input non-border
        obj_color = 0
        r_min, c_min, r_max, c_max = 100, 100, -1, -1
        for r in range(rows - 1):
            for c in range(cols - 1):
                if grid[r, c] != 0:
                    obj_color = grid[r, c]
                    r_min = min(r_min, r); c_min = min(c_min, c)
                    r_max = max(r_max, r); c_max = max(c_max, c)
        
        if obj_color == 0: return None
        
        # Object bounds in output
        or1, oc1 = r_min * F, c_min * F
        or2, oc2 = (r_max + 1) * F - 1, (c_max + 1) * F - 1
        
        # Target diagonals
        targets = [or1 - oc1, or2 - oc2, or1 + oc2, or2 + oc1]
        
        for r in range(nb_rows):
            for c in range(nb_cols):
                # Outside object rectangle
                if not (or1 <= r <= or2 and oc1 <= c <= oc2):
                    if (r - c) in [targets[0], targets[1]] or (r + c) in [targets[2], targets[3]]:
                        output[r, c] = 2
                                
        return output

    test_preds = []
    for inp, out in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out):
            return None
            
    for inp in solver.test_in:
        pred = run_single(inp)
        if pred is None: test_preds.append(np.array(inp))
        else: test_preds.append(pred)
        
    return test_preds
