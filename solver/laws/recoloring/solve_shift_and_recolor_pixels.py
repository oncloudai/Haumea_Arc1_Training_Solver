import numpy as np
from typing import List, Optional

def solve_shift_and_recolor_pixels(solver) -> Optional[List[np.ndarray]]:
    """
    Finds all pixels of a certain color, shifts them by a fixed vector, 
    and changes them to a new color.
    In task a79310a0: Color 8 (Azure) shifts (1, 0) and becomes Color 2 (Red).
    """
    def apply_logic(input_grid):
        input_grid = np.array(input_grid)
        rows, cols = input_grid.shape
        output_grid = np.zeros_like(input_grid)
        
        # Find all non-background pixels
        unique, counts = np.unique(input_grid, return_counts=True)
        bg_color = int(unique[np.argmax(counts)])
        
        active_colors = [int(c) for c in unique if c != bg_color]
        if not active_colors: return input_grid
        
        # We need to find the source color, shift, and target color.
        # Since this is a law, we'll try to infer it from the first pair.
        return None # To be filled by inferred logic if possible, or specialized

    # Specialized logic for a79310a0 style:
    def try_specific_params(inp, out):
        # Find all (r, c, color_in, dr, dc, color_out) that match
        # This is complex to generalize perfectly, so let's use the known rule
        # and see if it generalizes by trying to find it.
        
        # Let's just implement the specific known rule for this task type
        # but named generically.
        unique_in = np.unique(inp)
        unique_out = np.unique(out)
        
        c_in_candidates = [c for c in unique_in if c != 0]
        c_out_candidates = [c for c in unique_out if c != 0]
        
        for c_in in c_in_candidates:
            for c_out in c_out_candidates:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        
                        # Test this rule
                        test_out = np.zeros_like(inp)
                        coords = np.argwhere(inp == c_in)
                        for r, c in coords:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1]:
                                test_out[nr, nc] = c_out
                        
                        if np.array_equal(test_out, out):
                            return (c_in, c_out, dr, dc)
        return None

    # Infer params from first pair
    params = try_specific_params(solver.train_in[0], solver.train_out[0])
    if params is None: return None
    c_in, c_out, dr, dc = params
    
    def apply(grid):
        grid = np.array(grid)
        out = np.zeros_like(grid)
        coords = np.argwhere(grid == c_in)
        for r, c in coords:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                out[nr, nc] = c_out
        return out

    for inp, out_expected in solver.pairs:
        if not np.array_equal(apply(inp), out_expected):
            return None
            
    return [apply(ti) for ti in solver.test_in]
