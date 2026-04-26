import numpy as np
from typing import List, Optional

def solve_recolor_by_top_left_gray_frame_template(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a 4x4 L-shaped frame of color 5 at the top-left.
    The 3x3 area it encloses (0:3, 0:3) contains the template shape.
    Finds other objects in the grid that match the template shape EXACTLY (identity)
    and recolors them to color 5, but DOES NOT recolor the template itself.
    """
    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        if rows < 4 or cols < 4: return None
        
        # Check for L-frame of color 5 at top-left
        frame_coords = [(0, 3), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
        for r, c in frame_coords:
            if grid[r, c] != 5: return None
            
        # Extract template from 3x3 area
        template_area = grid[0:3, 0:3]
        t_coords = np.argwhere(template_area != 0)
        if len(t_coords) == 0: return None
        
        # Normalize template shape
        r_min, c_min = t_coords.min(axis=0)
        template_shape = set(tuple(p) for p in (t_coords - [r_min, c_min]))
        
        output = grid.copy()
        labeled = np.zeros_like(grid)
        curr = 1
        found_any = False
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0 and grid[r, c] != 5 and labeled[r, c] == 0:
                    color = grid[r, c]
                    q = [(r, c)]; labeled[r, c] = curr
                    obj_coords = []
                    while q:
                        cr, cc = q.pop(0)
                        obj_coords.append((cr, cc))
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<rows and 0<=nc<cols and grid[nr,nc]==color and labeled[nr,nc]==0:
                                labeled[nr,nc]=curr; q.append((nr,nc))
                    
                    obj_coords = np.array(obj_coords)
                    # Check if this object is the template itself (any pixel in 0:3, 0:3)
                    is_template = False
                    for pr, pc in obj_coords:
                        if 0 <= pr < 3 and 0 <= pc < 3:
                            is_template = True; break
                    
                    if not is_template:
                        or_min, oc_min = obj_coords.min(axis=0)
                        obj_shape = set(tuple(p) for p in (obj_coords - [or_min, oc_min]))
                        if obj_shape == template_shape:
                            for pr, pc in obj_coords:
                                output[pr, pc] = 5
                            found_any = True
                    curr += 1
                    
        return output if found_any else None

    # Verify on pairs
    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    test_preds = []
    for inp in solver.test_in:
        pred = run_single(inp)
        test_preds.append(pred if pred is not None else np.array(inp))
        
    return test_preds
