import numpy as np
from typing import List, Optional

def solve_extract_major_color_bbox_1x1(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a 'major' color that is the majority in its own bounding box.
    If multiple major colors are found, it sorts them by noise count.
    Returns the major color as a 1x1 grid.
    In task de1cd16c: returns the major color with highest noise count.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        unique_colors = np.unique(grid)
        
        candidates = []
        for c in unique_colors:
            if c == 0: continue
            
            # Find bounding box
            coords = np.argwhere(grid == c)
            if len(coords) == 0: continue
                
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            box = grid[r_min:r_max+1, c_min:c_max+1]
            
            pixels_count = len(coords)
            total_in_box = box.size
            noise_count = total_in_box - pixels_count
            
            # A color is major if it's the majority in its own bounding box
            if pixels_count > noise_count:
                candidates.append((c, noise_count))
                
        if not candidates: return None
            
        # Sort by noise count descending (as in training)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return np.array([[int(candidates[0][0])]])

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
