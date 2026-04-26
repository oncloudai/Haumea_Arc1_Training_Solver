
import numpy as np
from typing import List, Optional

def solve_expand_shape_by_cross_template(solver) -> Optional[List[np.ndarray]]:
    """
    Find two objects: one is a template cross with a marker (color 1), 
    the other is a target shape. Replicate the target shape at each 
    relative position defined by the template's non-zero pixels 
    relative to its marker.
    """
    def get_objects(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        labeled = np.zeros_like(grid)
        curr = 1
        objects = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0 and labeled[r, c] == 0:
                    obj_coords = []
                    q = [(r, c)]; labeled[r, c] = curr
                    color = grid[r, c]
                    while q:
                        cr, cc = q.pop(0)
                        obj_coords.append((cr, cc))
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and labeled[nr, nc] == 0:
                                if grid[nr, nc] != 0:
                                    labeled[nr, nc] = curr; q.append((nr, nc))
                    objects.append(obj_coords)
                    curr += 1
        return objects

    def apply_logic(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        objects = get_objects(grid)
        if len(objects) < 2: return None
        
        template_obj = None
        target_obj = None
        marker_color = 1
        
        for obj in objects:
            if any(grid[r, c] == marker_color for r, c in obj):
                template_obj = obj
            else:
                target_obj = obj
                
        if template_obj is None or target_obj is None: return None
        
        # Find marker in template
        marker_pos = [pos for pos in template_obj if grid[pos] == marker_color][0]
        # Template relative shifts (excluding marker itself if it's just a pointer)
        # In 57aa92db, the template is a cross.
        shifts = [(r - marker_pos[0], c - marker_pos[1]) for r, c in template_obj if grid[r, c] != marker_color]
        
        out = grid.copy()
        # Find target shape's color and marker (if it has one to be replaced)
        target_colors = [grid[pos] for pos in target_obj if grid[pos] != marker_color]
        if not target_colors: return None
        
        # Clear original marker in target if it was there? No, usually not.
        
        for dr, dc in shifts:
            for tr, tc in target_obj:
                nr, nc = tr + dr, tc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    out[nr, nc] = grid[tr, tc]
        
        # Remove the marker from the final output template position?
        # Looking at Ex 0, the marker (1) is STILL THERE in the output.
        # But wait, the shape itself might have 1s? No.
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
