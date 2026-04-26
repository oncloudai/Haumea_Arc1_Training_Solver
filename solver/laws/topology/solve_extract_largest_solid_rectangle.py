import numpy as np
from typing import List, Optional

def solve_extract_largest_solid_rectangle(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies the largest perfect solid rectangle of a single color in the input.
    The output grid is the same size as the input, but with only this rectangle preserved.
    Everything else is set to zero.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        H, W = grid.shape
        max_rect_area = -1
        final_rect = None
        
        for color in range(1, 10):
            color_mask = (grid == color).astype(int)
            heights = np.zeros(W, dtype=int)
            for r in range(H):
                for c in range(W):
                    if color_mask[r, c] == 1:
                        heights[c] += 1
                    else:
                        heights[c] = 0
                
                # Find largest rectangle in histogram `heights`
                stack = []
                # Append a 0 height to ensure all rectangles are processed
                temp_heights = list(heights) + [0]
                for i, h_val in enumerate(temp_heights):
                    while stack and temp_heights[stack[-1]] >= h_val:
                        H_rect = temp_heights[stack.pop()]
                        W_rect = i if not stack else i - stack[-1] - 1
                        if H_rect * W_rect > max_rect_area:
                            max_rect_area = H_rect * W_rect
                            # color, r1, r2, c1, c2
                            final_rect = (color, r - H_rect + 1, r, i - W_rect, i - 1)
                    stack.append(i)
        
        out = np.zeros_like(grid)
        if final_rect:
            c, r1, r2, c1, c2 = final_rect
            out[r1:r2+1, c1:c2+1] = c
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
