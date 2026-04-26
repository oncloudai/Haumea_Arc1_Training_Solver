import numpy as np
from typing import List, Optional

def solve_denoise_by_symmetry(solver) -> Optional[List[np.ndarray]]:
    for bg_to_remove in range(10):
        consistent = True
        found_any = False
        for inp, out in solver.pairs:
            h, w = inp.shape
            if inp.shape != out.shape: consistent = False; break
            
            # Check for global symmetry in output
            h_sym = np.array_equal(out, np.flipud(out))
            v_sym = np.array_equal(out, np.fliplr(out))
            if not h_sym and not v_sym: consistent = False; break
            
            # Try to recover output from input by using symmetry to fill pixels of color bg_to_remove
            pred = inp.copy()
            # If a pixel is bg_to_remove, find its symmetric counterparts and take the non-bg value
            for r in range(h):
                for c in range(w):
                    if inp[r, c] == bg_to_remove:
                        options = []
                        if h_sym: options.append(inp[h - 1 - r, c])
                        if v_sym: options.append(inp[r, w - 1 - c])
                        if h_sym and v_sym: options.append(inp[h - 1 - r, w - 1 - c])
                        
                        valid_options = [o for o in options if o != bg_to_remove]
                        if valid_options:
                            # Heuristic: most frequent valid option? Or just any?
                            pred[r, c] = valid_options[0]
            
            if not np.array_equal(pred, out):
                # Maybe the noise was DIFFERENT pixels?
                # Actually, in b8825c91, it seems color 4 is the noise.
                consistent = False; break
            found_any = True
            
        if consistent and found_any:
            def process(grid):
                h, w = grid.shape
                # Detect symmetry from first train output (assuming same for all)
                out0 = solver.train_out[0]
                h_sym = np.array_equal(out0, np.flipud(out0))
                v_sym = np.array_equal(out0, np.fliplr(out0))
                res = grid.copy()
                for r in range(h):
                    for c in range(w):
                        if grid[r, c] == bg_to_remove:
                            options = []
                            if h_sym: options.append(grid[h - 1 - r, c])
                            if v_sym: options.append(grid[r, w - 1 - c])
                            if h_sym and v_sym: options.append(grid[h - 1 - r, w - 1 - c])
                            valid = [o for o in options if o != bg_to_remove]
                            if valid: res[r, c] = valid[0]
                return res
            return [process(ti) for ti in solver.test_in]
    return None
