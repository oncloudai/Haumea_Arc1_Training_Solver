import numpy as np
from typing import List, Optional

def solve_triangle_wave_pixel_propagation(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a starting pixel (usually color 1) and propagates it upwards 
    in a triangle wave pattern with a background fill (usually color 8).
    Formula: col(r) = (W-1) - abs(((h-1-r) + offset) % (2*(W-1)) - (W-1))
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # Find the starting position and color of a single pixel
        coords = np.argwhere(grid != 0)
        # If there's more than one color or many pixels, this might not be the right law
        if len(coords) == 0: return None
        
        start_r, start_c = coords[0]
        color = grid[start_r, start_c]
        
        # Usually background is 8 in these tasks
        unique, counts = np.unique(grid, return_counts=True)
        # But we'll infer it from the output of the first pair
        return None

    def try_infer_bg_and_color(inp, out):
        coords_in = np.argwhere(inp != 0)
        if len(coords_in) != 1: return None
        r0, c0 = coords_in[0]
        color = inp[r0, c0]
        
        # Most common color in output that isn't the propagated color
        unique_out, counts_out = np.unique(out, return_counts=True)
        bg_out = -1
        for c in unique_out[np.argsort(counts_out)[::-1]]:
            if c != color:
                bg_out = c
                break
        if bg_out == -1: return None
        return color, bg_out

    if not solver.pairs: return None
    params = try_infer_bg_and_color(solver.train_in[0], solver.train_out[0])
    if params is None: return None
    color, bg_color = params
    
    def apply(grid):
        h, w = grid.shape
        if w == 1:
            out = np.full((h, w), color)
            return out
            
        coords = np.argwhere(grid == color)
        if len(coords) == 0: return np.full((h, w), bg_color)
        start_r, start_c = coords[0]
        
        out = np.full((h, w), bg_color)
        period = 2 * (w - 1)
        # offset such that at r = start_r, col = start_c
        # (w-1) - abs(( (h-1-start_r) + offset ) % period - (w-1)) = start_c
        # In ARC, start_r is usually h-1
        offset = start_c # Simplified, assuming start_r = h-1 and increasing
        
        for r in range(h - 1, -1, -1):
            i = (h - 1) - r
            curr_col = (w - 1) - abs(((i + offset) % period) - (w - 1))
            if 0 <= curr_col < w:
                out[r, curr_col] = color
        return out

    for inp, out_expected in solver.pairs:
        if not np.array_equal(apply(inp), out_expected):
            return None
            
    return [apply(ti) for ti in solver.test_in]
