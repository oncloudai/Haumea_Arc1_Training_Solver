
import numpy as np
from typing import List, Optional

def solve_extract_shape_with_marker_bbox(solver) -> Optional[List[np.ndarray]]:
    """
    Find all 4-connected components. Identify the one containing a specific marker (color 8).
    Extract that component (with the marker replaced by the component's color) 
    within its bounding box.
    """
    def apply_logic(input_grid, marker_color=8):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        labeled = np.zeros_like(grid)
        curr = 1
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0 and labeled[r, c] == 0:
                    q = [(r, c)]; labeled[r, c] = curr
                    color = grid[r, c]
                    while q:
                        cr, cc = q.pop(0)
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and labeled[nr, nc] == 0:
                                # Marker 8 can be part of any shape
                                if grid[nr, nc] != 0:
                                    # If it's the marker, it belongs to this shape
                                    if grid[nr, nc] == marker_color or color == marker_color or grid[nr, nc] == color:
                                        labeled[nr, nc] = curr; q.append((nr, nc))
                    curr += 1
        
        for i in range(1, curr):
            coords = np.argwhere(labeled == i)
            is_marker_present = any(grid[r, c] == marker_color for r, c in coords)
            if is_marker_present:
                # Find the dominant color in this shape (other than marker)
                colors = [grid[r, c] for r, c in coords if grid[r, c] != marker_color]
                if not colors: continue
                shape_color = max(set(colors), key=colors.count)
                
                r_min, c_min = coords.min(axis=0)
                r_max, c_max = coords.max(axis=0)
                
                # The output in the examples seems to be always 3x3
                # Let's check if we should return 3x3 or bbox
                res = np.zeros((r_max - r_min + 1, c_max - c_min + 1), dtype=int)
                for r, c in coords:
                    if grid[r, c] != marker_color:
                        res[r - r_min, c - c_min] = grid[r, c]
                    else:
                        # Replace marker with shape color or 0? 
                        # In Ex 0: 4 was the shape, 8 was at (1, 10). output[1, 1] is 4.
                        # So marker is replaced by shape color.
                        res[r - r_min, c - c_min] = shape_color
                return res, True
        return None, False

    for inp, out in solver.pairs:
        pred, found = apply_logic(inp)
        if not found or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res, _ = apply_logic(ti)
        results.append(res)
    return results
