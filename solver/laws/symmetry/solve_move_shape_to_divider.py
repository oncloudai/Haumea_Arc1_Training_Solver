
import numpy as np
from typing import List, Optional

def solve_move_shape_to_divider(solver) -> Optional[List[np.ndarray]]:
    """
    Find a shape made of 3s and a barrier line of 2s.
    Move the shape to be adjacent to the 2-line.
    Place an 8-line at (original_max_dist + 1 - shift) from the divider.
    """
    def apply_logic(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        output = np.zeros_like(grid)
        
        # Find the 2-line
        row_2s = [r for r in range(rows) if np.all(grid[r, :] == 2)]
        col_2s = [c for c in range(cols) if np.all(grid[:, c] == 2)]
        
        r3, c3 = np.where(grid == 3)
        if len(r3) == 0: return grid, False

        if row_2s:
            r2 = row_2s[0]
            output[r2, :] = 2
            dists = np.abs(r3 - r2)
            min_d = np.min(dists)
            max_d = np.max(dists)
            shift = min_d - 1
            if shift < 0: return grid, False # Already adjacent or on the line
            
            new_dists = dists - shift
            line_8_dist = max_d + 1 - shift
            
            direction = 1 if r3[0] > r2 else -1
            for i in range(len(r3)):
                nr = r2 + direction * new_dists[i]
                if 0 <= nr < rows:
                    output[nr, c3[i]] = 3
            
            n8r = r2 + direction * line_8_dist
            if 0 <= n8r < rows:
                output[n8r, :] = 8
            return output, True
            
        elif col_2s:
            c2 = col_2s[0]
            output[:, c2] = 2
            dists = np.abs(c3 - c2)
            min_d = np.min(dists)
            max_d = np.max(dists)
            shift = min_d - 1
            if shift < 0: return grid, False
            
            new_dists = dists - shift
            line_8_dist = max_d + 1 - shift
            
            direction = 1 if c3[0] > c2 else -1
            for i in range(len(c3)):
                nc = c2 + direction * new_dists[i]
                if 0 <= nc < cols:
                    output[r3[i], nc] = 3
            
            n8c = c2 + direction * line_8_dist
            if 0 <= n8c < cols:
                output[:, n8c] = 8
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
