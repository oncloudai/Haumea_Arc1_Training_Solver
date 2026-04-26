import numpy as np
from typing import List, Optional

def solve_connect_points_l_shape(solver) -> Optional[List[np.ndarray]]:
    # Try all pairs of colors for start, end, and fill
    for start_color in range(1, 10):
        for end_color in range(1, 10):
            if start_color == end_color: continue
            for fill_color in range(1, 10):
                if fill_color == start_color or fill_color == end_color: continue
                
                consistent = True
                found_any = False
                
                for inp, out in solver.pairs:
                    start_coords = np.argwhere(inp == start_color)
                    end_coords = np.argwhere(inp == end_color)
                    
                    if len(start_coords) != 1 or len(end_coords) != 1:
                        consistent = False; break
                    
                    r1, c1 = start_coords[0]
                    r2, c2 = end_coords[0]
                    
                    pred = inp.copy()
                    
                    # Horizontal part: from (r1, c1) to (r1, c2)
                    c_step = 1 if c2 > c1 else -1
                    for c in range(c1 + c_step, c2 + c_step, c_step):
                        if (r1, c) != (r2, c2):
                            if pred[r1, c] == 0 or pred[r1, c] == fill_color:
                                pred[r1, c] = fill_color
                            elif pred[r1, c] == start_color or pred[r1, c] == end_color:
                                pass # allow overlap with endpoints
                            else:
                                # If it hits something else, maybe it's still okay?
                                # But let's be strict for now.
                                pred[r1, c] = fill_color
                    
                    # Vertical part: from (r1, c2) to (r2, c2)
                    r_step = 1 if r2 > r1 else -1
                    for r in range(r1 + r_step, r2 + r_step, r_step):
                        if (r, c2) != (r2, c2):
                            if pred[r, c2] == 0 or pred[r, c2] == fill_color:
                                pred[r, c2] = fill_color
                            else:
                                pred[r, c2] = fill_color
                                
                    if not np.array_equal(pred, out):
                        consistent = False; break
                    found_any = True
                    
                if consistent and found_any:
                    results = []
                    for ti in solver.test_in:
                        start_coords = np.argwhere(ti == start_color)
                        end_coords = np.argwhere(ti == end_color)
                        if len(start_coords) != 1 or len(end_coords) != 1:
                            results.append(ti.copy()); continue
                        
                        r1, c1 = start_coords[0]
                        r2, c2 = end_coords[0]
                        res = ti.copy()
                        
                        c_step = 1 if c2 > c1 else -1
                        for c in range(c1 + c_step, c2 + c_step, c_step):
                            if (r1, c) != (r2, c2): res[r1, c] = fill_color
                        
                        r_step = 1 if r2 > r1 else -1
                        for r in range(r1 + r_step, r2 + r_step, r_step):
                            if (r, c2) != (r2, c2): res[r, c2] = fill_color
                            
                        results.append(res)
                    return results
    return None
