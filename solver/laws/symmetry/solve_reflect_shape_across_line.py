
import numpy as np
from typing import List, Optional

def solve_reflect_shape_across_line(solver) -> Optional[List[np.ndarray]]:
    """
    Find a shape made of 3s and a barrier line of 2s.
    Move the shape to be adjacent to the 2-line on the opposite side.
    Place an 8-line where the shape previously was (at the far boundary).
    """
    def apply_logic(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        output = np.zeros_like(grid)
        
        # Find the 2-line (either a full row or a full column)
        row_2s = [r for r in range(rows) if np.all(grid[r, :] == 2)]
        col_2s = [c for c in range(cols) if np.all(grid[:, c] == 2)]
        
        if row_2s:
            r2 = row_2s[0]
            output[r2, :] = 2
            # Find the shape of 3s
            r3, c3 = np.where(grid == 3)
            if len(r3) == 0: return grid, False
            
            # Original bounding box of 3s
            min_r, max_r = r3.min(), r3.max()
            min_c, max_c = c3.min(), c3.max()
            shape_h = max_r - min_r + 1
            
            if min_r > r2: # Shape is below the line
                # Move shape to just above the line
                new_max_r = r2 - 1
                new_min_r = new_max_r - (shape_h - 1)
                for r, c in zip(r3, c3):
                    output[new_min_r + (r - min_r), c] = 3
                # Place 8-line at the original bottom boundary of the shape
                output[max_r, :] = 8
            else: # Shape is above the line
                # Move shape to just below the line
                new_min_r = r2 + 1
                new_max_r = new_min_r + (shape_h - 1)
                for r, c in zip(r3, c3):
                    output[new_min_r + (r - min_r), c] = 3
                # Place 8-line at the original top boundary of the shape
                output[min_r, :] = 8
            return output, True
            
        elif col_2s:
            c2 = col_2s[0]
            output[:, c2] = 2
            # Find the shape of 3s
            r3, c3 = np.where(grid == 3)
            if len(r3) == 0: return grid, False
            
            min_r, max_r = r3.min(), r3.max()
            min_c, max_c = c3.min(), c3.max()
            shape_w = max_c - min_c + 1
            
            if min_c > c2: # Shape is to the right
                # Move shape to just left of the line
                new_max_c = c2 - 1
                new_min_c = new_max_c - (shape_w - 1)
                for r, c in zip(r3, c3):
                    output[r, new_min_c + (c - min_c)] = 3
                # Place 8-line at the original right boundary of the shape
                output[:, max_c] = 8
            else: # Shape is to the left
                # Move shape to just right of the line
                new_min_c = c2 + 1
                new_max_c = new_min_c + (shape_w - 1)
                for r, c in zip(r3, c3):
                    output[r, new_min_c + (c - min_c)] = 3
                # Place 8-line at the original left boundary of the shape
                output[:, min_c] = 8
            return output, True
            
        return grid, False

    for inp, out in solver.pairs:
        pred, found = apply_logic(inp)
        if not found or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res, _ = apply_logic(ti)
        results.append(res)
    return results
