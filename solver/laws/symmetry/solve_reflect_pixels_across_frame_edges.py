import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_reflect_pixels_across_frame_edges(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a larger frame-like object in the input. Smaller pixels (usually size 1)
    are reflected across the edges of this frame. 
    If a pixel is at distance D from an edge inside the frame, it is moved to 
    distance D from the opposite edge outside the frame (or vice versa).
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # 1. Find the large object (the frame)
        frame_bbox = None
        frame_mask = np.zeros_like(grid, dtype=bool)
        # Iterate colors to find the most likely frame (size > 1)
        for color in range(1, 10):
            mask = (grid == color).astype(int)
            labeled, num_features = label(mask)
            for i in range(1, num_features + 1):
                comp_mask = (labeled == i)
                if np.sum(comp_mask) > 1:
                    # Found a potential frame
                    frame_mask |= comp_mask
                    rows, cols = np.where(comp_mask)
                    frame_bbox = (rows.min(), rows.max(), cols.min(), cols.max())
                    break
            if frame_bbox: break
            
        if not frame_bbox: return None
        
        r1, r2, c1, c2 = frame_bbox
        out = np.zeros_like(grid)
        out[frame_mask] = grid[frame_mask]
        
        # 2. Reflect pixels not in the frame
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0 and not frame_mask[r, c]:
                    dr_top = r - r1
                    dr_bot = r2 - r
                    dc_left = c - c1
                    dc_right = c2 - c
                    
                    # Vertical flip across frame
                    if dr_top > dr_bot:
                        nr = r1 - dr_bot
                    else:
                        nr = r2 + dr_top
                        
                    # Horizontal flip across frame
                    if dc_left > dc_right:
                        nc = c1 - dc_right
                    else:
                        nc = c2 + dc_left
                        
                    if 0 <= nr < h and 0 <= nc < w:
                        out[nr, nc] = grid[r, c]
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
