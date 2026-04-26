import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_extract_major_region_with_seeds(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies the 'major' component (Color A) whose bounding box defines the output.
    Identifies 'seed' pixels (non-zero, non-Color A) within this bounding box.
    For each seed at relative position (r, c), fills row r and column c 
    of the output grid with the seed's color.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # 1. Find the major color A and its bbox
        # We look for the color that has the largest component by pixel count.
        best_color = -1
        best_bbox = None
        max_pix = -1
        
        for color in range(1, 10):
            mask = (grid == color).astype(int)
            labeled, num_f = label(mask, structure=np.ones((3,3)))
            for i in range(1, num_f + 1):
                comp = (labeled == i)
                num_pix = np.sum(comp)
                if num_pix > max_pix:
                    max_pix = num_pix
                    best_color = color
                    rows, cols = np.where(comp)
                    best_bbox = (rows.min(), rows.max(), cols.min(), cols.max())
                    
        if best_bbox is None: return None
        r1, r2, c1, c2 = best_bbox
        sh, sw = r2 - r1 + 1, c2 - c1 + 1
        
        # Initialize output with major color
        out = np.full((sh, sw), best_color, dtype=int)
        
        # 2. Identify seeds and their colors
        # A seed is any non-zero, non-Color A pixel within the bbox
        seeds = []
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                val = grid[r, c]
                if val != 0 and val != best_color:
                    seeds.append((r - r1, c - c1, val))
                    
        if not seeds:
            # If no colored seeds, check for 0s as seeds? 
            # (In some tasks, 0s might be the seeds, but let's stick to colors first).
            pass
            
        # 3. Apply seed crosses
        # Sort seeds to have a consistent overlap rule (e.g. larger color value wins)
        seeds.sort(key=lambda x: x[2])
        
        for sr, sc, s_color in seeds:
            out[sr, :] = s_color
            out[:, sc] = s_color
            
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
