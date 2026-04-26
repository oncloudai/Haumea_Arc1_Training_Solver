import numpy as np
from typing import List, Optional

def solve_object_move_to_marker_center(solver) -> Optional[List[np.ndarray]]:
    # Try all pairs of colors for object and markers
    for obj_color in range(1, 10):
        for marker_color in range(1, 10):
            if obj_color == marker_color: continue
            
            consistent = True
            found_any = False
            
            for inp, out in solver.pairs:
                obj_coords = np.argwhere(inp == obj_color)
                marker_coords = np.argwhere(inp == marker_color)
                
                if len(obj_coords) == 0 or len(marker_coords) == 0:
                    consistent = False; break
                
                r_min_m, c_min_m = marker_coords.min(axis=0)
                r_max_m, c_max_m = marker_coords.max(axis=0)
                m_center_r, m_center_c = (r_min_m + r_max_m) / 2, (c_min_m + c_max_m) / 2
                
                r_min_o, c_min_o = obj_coords.min(axis=0)
                r_max_o, c_max_o = obj_coords.max(axis=0)
                o_center_r, o_center_c = (r_min_o + r_max_o) / 2, (c_min_o + c_max_o) / 2
                
                dr, dc = int(m_center_r - o_center_r), int(m_center_c - o_center_c)
                
                pred = inp.copy()
                # Clear object
                for r, c in obj_coords:
                    if pred[r, c] == obj_color:
                        pred[r, c] = 0
                
                # Place object at new location
                possible = True
                for r, c in obj_coords:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < pred.shape[0] and 0 <= nc < pred.shape[1]:
                        pred[nr, nc] = obj_color
                    else:
                        possible = False; break
                
                if not possible or not np.array_equal(pred, out):
                    consistent = False; break
                found_any = True
                
            if consistent and found_any:
                results = []
                for ti in solver.test_in:
                    obj_coords = np.argwhere(ti == obj_color)
                    marker_coords = np.argwhere(ti == marker_color)
                    if len(obj_coords) == 0 or len(marker_coords) == 0:
                        results.append(ti.copy()); continue
                        
                    r_min_m, c_min_m = marker_coords.min(axis=0)
                    r_max_m, c_max_m = marker_coords.max(axis=0)
                    m_center_r, m_center_c = (r_min_m + r_max_m) / 2, (c_min_m + c_max_m) / 2
                    
                    r_min_o, c_min_o = obj_coords.min(axis=0)
                    r_max_o, c_max_o = obj_coords.max(axis=0)
                    o_center_r, o_center_c = (r_min_o + r_max_o) / 2, (c_min_o + c_max_o) / 2
                    
                    dr, dc = int(m_center_r - o_center_r), int(m_center_c - o_center_c)
                    
                    res = ti.copy()
                    for r, c in obj_coords:
                        if res[r, c] == obj_color: res[r, c] = 0
                    for r, c in obj_coords:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < res.shape[0] and 0 <= nc < res.shape[1]:
                            res[nr, nc] = obj_color
                    results.append(res)
                return results
    return None
