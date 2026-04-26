
import numpy as np
from typing import List, Optional

def solve_mirrored_seed_by_divider(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a cross-shaped divider. Extracts the non-zero seed from one quadrant.
    Mirrors the seed horizontally and vertically to create a 2x2 tiling of quadrants.
    The non-zero pixels in the output are colored with the divider color.
    """
    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        
        divider_color = -1
        div_r, div_c = -1, -1
        
        # Find divider cross
        for r in range(rows):
            unique = np.unique(grid[r, :])
            if len(unique) == 1 and unique[0] != 0:
                color = unique[0]
                for c in range(cols):
                    unique_col = np.unique(grid[:, c])
                    if len(unique_col) == 1 and unique_col[0] == color:
                        divider_color = color
                        div_r, div_c = r, c
                        break
                if divider_color != -1: break
                
        if divider_color == -1: return None
        
        # Quadrants relative to divider
        quads = [
            grid[:div_r, :div_c],
            grid[:div_r, div_c+1:],
            grid[div_r+1:, :div_c],
            grid[div_r+1:, div_c+1:]
        ]
        
        seed = None
        for q in quads:
            if np.any(q != 0):
                seed = q
                break
        if seed is None: return None
        
        # Mirroring
        h_mirrored = np.concatenate([seed, np.fliplr(seed)], axis=1)
        full_mirrored = np.concatenate([h_mirrored, np.flipud(h_mirrored)], axis=0)
        
        # Recolor non-zero to divider color
        return np.where(full_mirrored != 0, divider_color, 0)

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
