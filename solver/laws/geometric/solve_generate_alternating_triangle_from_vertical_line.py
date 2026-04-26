import numpy as np
from typing import List, Optional

def solve_generate_alternating_triangle_from_vertical_line(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a vertical line of a specific color.
    Finds the bottom-most pixel of this line.
    Generates an upward-spreading triangle from that bottom-most pixel.
    The triangle rows have alternating colors based on distance from the center column.
    """
    def try_infer_colors(inp, out):
        # Color 1 is the color of the vertical line in input
        unique_in = np.unique(inp)
        non_bg_in = unique_in[unique_in != 0]
        if len(non_bg_in) != 1: return None
        c1 = int(non_bg_in[0])
        
        # Color 2 is the other non-background color in the output triangle
        unique_out = np.unique(out)
        non_bg_out = [int(c) for c in unique_out if c != 0 and c != c1]
        if not non_bg_out: return None
        c2 = non_bg_out[0]
        return c1, c2

    if not solver.pairs: return None
    params = try_infer_colors(solver.train_in[0], solver.train_out[0])
    if params is None: return None
    c1, c2 = params

    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        coords = np.argwhere(grid == c1)
        if len(coords) == 0: return grid
        
        # Find column and bottom-most row
        c_mid = coords[0, 1]
        # Verify it's a vertical line in that column
        if not np.all(coords[:, 1] == c_mid): return grid
        
        r_end = coords[:, 0].max()
        out = np.zeros_like(grid)
        
        for r in range(r_end + 1):
            d = r_end - r
            for k in range(d + 1):
                color = c1 if k % 2 == 0 else c2
                if c_mid - k >= 0:
                    out[r, c_mid - k] = color
                if c_mid + k < cols:
                    out[r, c_mid + k] = color
        return out

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
