import numpy as np
from typing import List, Optional

def solve_extract_rotated_region_by_color_3(solver) -> Optional[List[np.ndarray]]:
    """
    Finds the rectangular bounding box of all pixels with color 3.
    Calculates the 180-degree rotated position of this bounding box in the grid.
    Extracts the region at that rotated position and rotates it by 180 degrees.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # Find the rectangular block of 3's
        coords = np.argwhere(grid == 3)
        if len(coords) == 0: return None
        
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        
        # Calculate the 180-degree rotated position
        # For point (r, c) in hxw grid, 180-rotation gives (h-1-r, w-1-c)
        new_r_min = h - 1 - r_max
        new_r_max = h - 1 - r_min
        new_c_min = w - 1 - c_max
        new_c_max = w - 1 - c_min
        
        # Extract the region at the rotated position
        if not (0 <= new_r_min < h and 0 <= new_r_max < h and 
                0 <= new_c_min < w and 0 <= new_c_max < w):
            return None
            
        extracted = grid[new_r_min:new_r_max+1, new_c_min:new_c_max+1]
        
        # Apply 180-degree rotation to the extracted region
        result = np.rot90(extracted, 2)
        return result

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
