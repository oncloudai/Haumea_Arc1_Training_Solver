import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_flip_solid_rectangle_in_bbox(solver) -> Optional[List[np.ndarray]]:
    """
    For each connected component of a color:
    1. Find the largest solid rectangle (SR) within it.
    2. Find the bounding box (BB) of the entire component.
    3. Place the SR at the opposite side of its original position within the BB.
    (If SR was at top of BB, move to bottom; if at left, move to right, and vice-versa).
    If SR already fills the BB, it stays in place.
    """
    def find_largest_rect(mask):
        H, W = mask.shape
        max_area = 0
        best_rect = None # r1, r2, c1, c2
        heights = np.zeros(W, dtype=int)
        for r in range(H):
            for c in range(W):
                if mask[r, c] == 1:
                    heights[c] += 1
                else:
                    heights[c] = 0
            stack = []
            temp_heights = list(heights) + [0]
            for i, h in enumerate(temp_heights):
                while stack and temp_heights[stack[-1]] >= h:
                    H_rect = temp_heights[stack.pop()]
                    W_rect = i if not stack else i - stack[-1] - 1
                    if H_rect * W_rect > max_area:
                        max_area = H_rect * W_rect
                        best_rect = (r - H_rect + 1, r, i - W_rect, i - 1)
                stack.append(i)
        return best_rect

    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        out = np.zeros_like(grid)
        
        for color in range(1, 10):
            mask = (grid == color).astype(int)
            labeled, num_features = label(mask)
            for i in range(1, num_features + 1):
                comp_mask = (labeled == i).astype(int)
                # BB of the whole component
                rows, cols = np.where(comp_mask == 1)
                br1, br2 = rows.min(), rows.max()
                bc1, bc2 = cols.min(), cols.max()
                
                # Largest SR within the component
                sr = find_largest_rect(comp_mask)
                if not sr: continue
                sr1, sr2, sc1, sc2 = sr
                sr_h = sr2 - sr1 + 1
                sr_w = sc2 - sc1 + 1
                
                # Original offsets relative to BB
                top_offset = sr1 - br1
                bottom_offset = br2 - sr2
                left_offset = sc1 - bc1
                right_offset = bc2 - sc2
                
                # Flip: new_top = bottom_offset, new_left = right_offset
                new_sr1 = br1 + bottom_offset
                new_sc1 = bc1 + right_offset
                
                out[new_sr1:new_sr1+sr_h, new_sc1:new_sc1+sr_w] = color
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
