import numpy as np
from typing import List, Optional

def solve_block_move_one_step_on_path(solver) -> Optional[List[np.ndarray]]:
    def run_single(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        
        red_coords = np.argwhere(grid == 2)
        if len(red_coords) == 0: return None
        r_min, c_min = red_coords.min(axis=0)
        r_max, c_max = red_coords.max(axis=0)
        
        path_coords = np.argwhere(grid == 3)
        if len(path_coords) == 0: return None
        path = sorted([tuple(p) for p in path_coords])
        
        # Check all path points inside or touching the block
        current_idx = None
        for i, (pr, pc) in enumerate(path):
            if r_min <= pr <= r_max and c_min <= pc <= c_max:
                current_idx = i
                break
        
        if current_idx is None or current_idx + 1 >= len(path):
            return None
            
        new_p = path[current_idx + 1]
        old_p = path[current_idx]
        dr, dc = new_p[0] - old_p[0], new_p[1] - old_p[1]
        
        out = grid.copy()
        # Remove old reds
        for r, c in red_coords:
            out[r, c] = grid[r, c] # placeholder
        
        # We need to know what was under the red pixels
        # In this task, it's either 0 or 3.
        # Let's just clear reds to 0 first, then restore 3s if they were there.
        for r, c in red_coords:
            out[r, c] = 0
            
        # Re-place path pixels (they should not be deleted by moving block)
        for r, c in path_coords:
            out[r, c] = 3
            
        # Move reds
        for r, c in red_coords:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                out[nr, nc] = 2
            else:
                return None
        return out

    results = []
    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or pred.shape != out_expected.shape or not np.array_equal(pred, out_expected):
            return None
    
    for ti in solver.test_in:
        results.append(run_single(ti))
    return results
