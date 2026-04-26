import numpy as np
from typing import List, Optional

def solve_draw_3x3_frames_by_color_map(solver) -> Optional[List[np.ndarray]]:
    """
    For each single-pixel object of a specific color, draws a 3x3 hollow square
    of a corresponding color centered at that pixel's position.
    The mapping between input pixel color and output frame color is learned from training.
    """
    def get_mapping(grid_in, grid_out):
        grid_in = np.array(grid_in)
        grid_out = np.array(grid_out)
        h, w = grid_in.shape
        mapping = {}
        
        # Find all non-zero pixels in input
        rows, cols = np.where(grid_in != 0)
        for r, c in zip(rows, cols):
            val = grid_in[r, c]
            # Look at its 3x3 neighborhood in output (excluding the center)
            found_colors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if grid_out[nr, nc] != 0 and grid_out[nr, nc] != val:
                            found_colors.append(grid_out[nr, nc])
            if found_colors:
                # Assuming the frame is a single color
                mapping[val] = found_colors[0]
        return mapping

    # Accumulate mapping from all training pairs
    global_mapping = {}
    for inp, out in solver.pairs:
        m = get_mapping(inp, out)
        global_mapping.update(m)
        
    if not global_mapping:
        return None

    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        out = grid.copy()
        for r in range(h):
            for c in range(w):
                val = grid[r, c]
                if val in global_mapping:
                    t = global_mapping[val]
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                out[nr, nc] = t
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
