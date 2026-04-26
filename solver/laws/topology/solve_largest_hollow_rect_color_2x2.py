
import numpy as np
from typing import List, Optional

def solve_largest_hollow_rect_color_2x2(solver) -> Optional[List[np.ndarray]]:
    """
    Finds all hollow rectangular frames in the grid.
    Identifies the one with the largest hole area.
    Outputs a 2x2 grid of that rectangle's color.
    """
    def get_largest_hole_color(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        unique_colors = np.unique(grid)
        unique_colors = unique_colors[unique_colors != 0]
        
        best_color = None
        max_area = -1
        
        for color in unique_colors:
            coords = np.argwhere(grid == color)
            if len(coords) < 4: continue
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            h, w = r_max - r_min + 1, c_max - c_min + 1
            
            if h < 3 or w < 3: continue
            
            # Check if it's a hollow rectangle of this color
            is_rect = True
            for r in range(r_min, r_max + 1):
                if grid[r, c_min] != color or grid[r, c_max] != color:
                    is_rect = False; break
            if not is_rect: continue
            for c in range(c_min, c_max + 1):
                if grid[r_min, c] != color or grid[r_max, c] != color:
                    is_rect = False; break
            if not is_rect: continue
            
            # Hole dimensions
            hole_h, hole_w = h - 2, w - 2
            area = hole_h * hole_w
            
            if area > max_area:
                max_area = area
                best_color = color
            elif area == max_area and best_color is not None:
                # Tie-breaker: maybe smaller perimeter? Or just keep first?
                # For now, just keep first.
                pass
                
        return best_color

    test_preds = []
    # Verify on all training examples
    for inp, out in solver.pairs:
        color = get_largest_hole_color(inp)
        if color is None: return None
        pred = np.full((2, 2), color)
        if not np.array_equal(pred, out):
            return None
            
    # Apply to test inputs
    for inp in solver.test_in:
        color = get_largest_hole_color(inp)
        if color is None:
            # Fallback if no hollow rect found in test (should not happen if logic is correct)
            test_preds.append(np.zeros((2, 2), dtype=int))
        else:
            test_preds.append(np.full((2, 2), color))
            
    return test_preds
