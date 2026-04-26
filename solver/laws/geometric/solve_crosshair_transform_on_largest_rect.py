import numpy as np
from itertools import combinations
from typing import List, Optional

def solve_crosshair_transform_on_largest_rect(solver) -> Optional[List[np.ndarray]]:
    """
    Finds the largest pure rectangle consisting of exactly two colors in the input.
    Then performs a crosshair transform on it:
    - Finds the foreground color (least frequent color in the rectangle).
    - For each foreground pixel, draws a horizontal and vertical line through it 
      within the rectangle.
    - Sets all other pixels in the rectangle to the background color.
    """
    def find_largest_pure_rectangle_any_pair(grid):
        rows, cols = grid.shape
        max_area = 0
        best_rect_coords = None
        best_pair = None
        
        unique_colors = np.unique(grid).tolist()
        if 0 in unique_colors: unique_colors.remove(0)
        
        for c1, c2 in combinations(unique_colors, 2):
            is_pure = (grid == c1) | (grid == c2)
            
            heights = np.zeros(cols, dtype=int)
            for r in range(rows):
                for c in range(cols):
                    if is_pure[r, c]:
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
                        area = H_rect * W_rect
                        if area > max_area:
                            max_area = area
                            best_rect_coords = (r - H_rect + 1, r + 1, i - W_rect, i)
                            best_pair = (c1, c2)
                    stack.append(i)
        
        if best_rect_coords is None: return None, None
        r_start, r_end, c_start, c_end = best_rect_coords
        return grid[r_start:r_end, c_start:c_end].copy(), best_pair

    def crosshair_transform(rect, pair):
        if rect is None: return None
        c1, c2 = pair
        # Determine background vs crosshair color
        count1 = np.sum(rect == c1)
        count2 = np.sum(rect == c2)
        
        if count1 >= count2:
            bg_color, fg_color = c1, c2
        else:
            bg_color, fg_color = c2, c1
            
        output = np.full(rect.shape, bg_color, dtype=int)
        
        # Find all fg_color pixels in original rect
        coords = np.argwhere(rect == fg_color)
        for r, c in coords:
            output[r, :] = fg_color
            output[:, c] = fg_color
            
        return output

    def apply_logic(grid):
        grid = np.array(grid)
        rect, pair = find_largest_pure_rectangle_any_pair(grid)
        if rect is None: return None
        return crosshair_transform(rect, pair)

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
