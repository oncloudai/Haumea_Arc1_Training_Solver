
import numpy as np
from typing import List, Optional

def solve_cross_at_object_centers(solver) -> Optional[List[np.ndarray]]:
    """
    For each object of a specific color, find its center and draw a cross of another color through it.
    The cross only overwrites background pixels.
    """
    for obj_color in range(1, 10):
        for cross_color in range(1, 10):
            if obj_color == cross_color: continue
            consistent = True
            for inp, out in solver.pairs:
                res = inp.copy()
                bg_color = 0 # Or most frequent?
                # Find all obj_color pixels
                obj_pixels = np.argwhere(inp == obj_color)
                if len(obj_pixels) == 0: consistent = False; break
                
                # Group into objects
                labeled = np.zeros_like(inp)
                num_objects = 0
                for r, c in obj_pixels:
                    if labeled[r, c] == 0:
                        num_objects += 1
                        q = [(r, c)]
                        labeled[r, c] = num_objects
                        while q:
                            curr_r, curr_c = q.pop(0)
                            for dr in [-1, 0, 1]:
                                for dc in [-1, 0, 1]:
                                    if dr == 0 and dc == 0: continue
                                    nr, nc = curr_r + dr, curr_c + dc
                                    if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1] and inp[nr, nc] == obj_color and labeled[nr, nc] == 0:
                                        labeled[nr, nc] = num_objects
                                        q.append((nr, nc))
                
                centers = []
                for obj_id in range(1, num_objects + 1):
                    obj_coords = np.argwhere(labeled == obj_id)
                    r_min, r_max = obj_coords[:, 0].min(), obj_coords[:, 0].max()
                    c_min, c_max = obj_coords[:, 1].min(), obj_coords[:, 1].max()
                    centers.append(((r_min + r_max) // 2, (c_min + c_max) // 2))
                
                # We need to find the background color. 
                # Let's assume the pixels that are different in out vs inp (except markers) are the cross.
                # Actually, let's just use the bg_color from the input.
                # In 41e4d17e, the bg_color is 8.
                for bg_c in [0, 8]:
                    temp_res = inp.copy()
                    for rc, cc in centers:
                        # Draw vertical line
                        for r in range(inp.shape[0]):
                            if temp_res[r, cc] == bg_c: temp_res[r, cc] = cross_color
                        # Draw horizontal line
                        for c in range(inp.shape[1]):
                            if temp_res[rc, c] == bg_c or temp_res[rc, c] == cross_color:
                                temp_res[rc, c] = cross_color
                    if np.array_equal(temp_res, out):
                        res = temp_res
                        break
                else:
                    consistent = False; break
            
            if consistent:
                # Found the right parameters!
                results = []
                for ti in solver.test_in:
                    # Repeat the same process for test input
                    obj_pixels = np.argwhere(ti == obj_color)
                    labeled = np.zeros_like(ti)
                    num_objects = 0
                    for r, c in obj_pixels:
                        if labeled[r, c] == 0:
                            num_objects += 1
                            q = [(r, c)]
                            labeled[r, c] = num_objects
                            while q:
                                curr_r, curr_c = q.pop(0)
                                for dr in [-1, 0, 1]:
                                    for dc in [-1, 0, 1]:
                                        nr, nc = curr_r + dr, curr_c + dc
                                        if 0 <= nr < ti.shape[0] and 0 <= nc < ti.shape[1] and ti[nr, nc] == obj_color and labeled[nr, nc] == 0:
                                            labeled[nr, nc] = num_objects
                                            q.append((nr, nc))
                    centers = []
                    for obj_id in range(1, num_objects + 1):
                        obj_coords = np.argwhere(labeled == obj_id)
                        centers.append(((obj_coords[:,0].min() + obj_coords[:,0].max()) // 2, (obj_coords[:,1].min() + obj_coords[:,1].max()) // 2))
                    
                    res = ti.copy()
                    for rc, cc in centers:
                        for r in range(ti.shape[0]):
                            if res[r, cc] == bg_c: res[r, cc] = cross_color
                        for c in range(ti.shape[1]):
                            if res[rc, c] == bg_c or res[rc, c] == cross_color:
                                res[rc, c] = cross_color
                    results.append(res)
                return results
    return None
