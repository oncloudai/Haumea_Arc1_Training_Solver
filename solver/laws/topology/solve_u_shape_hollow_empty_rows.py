import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_u_shape_hollow_empty_rows(solver) -> Optional[List[np.ndarray]]:
    for bg_color in range(10):
        for container_color in range(10):
            if container_color == bg_color: continue
            for azure_color in range(10):
                if azure_color == bg_color or azure_color == container_color: continue
                
                consistent = True
                found_any = False
                for inp, out in solver.pairs:
                    if out.shape != (3, 3): consistent = False; break
                    
                    blobs = get_blobs(inp, bg_color)
                    containers = [b for b in blobs if b['color'] == container_color]
                    if not containers: consistent = False; break
                    
                    # Try each container blob
                    solved_this_pair = False
                    for c_blob in containers:
                        coords = c_blob['coords']
                        r_min, c_min = coords.min(axis=0)
                        r_max, c_max = coords.max(axis=0)
                        
                        hollow_rows = range(r_min, r_max)
                        hollow_cols = range(c_min + 1, c_max)
                        
                        if not hollow_rows or not hollow_cols: continue
                        
                        empty_rows_count = 0
                        for r in hollow_rows:
                            row_has_azure = False
                            for c in hollow_cols:
                                if inp[r, c] == azure_color:
                                    row_has_azure = True
                                    break
                            if not row_has_azure:
                                empty_rows_count += 1
                        
                        pred = np.zeros((3, 3), dtype=int)
                        fill_order = [(0,0), (0,1), (0,2), (1,2), (2,2), (2,1), (2,0), (1,0), (1,1)]
                        for i in range(min(empty_rows_count, 9)):
                            pr, pc = fill_order[i]
                            pred[pr, pc] = azure_color
                        
                        if np.array_equal(pred, out):
                            solved_this_pair = True; break
                    
                    if not solved_this_pair:
                        consistent = False; break
                    found_any = True
                
                if consistent and found_any:
                    results = []
                    for ti in solver.test_in:
                        blobs = get_blobs(ti, bg_color)
                        containers = [b for b in blobs if b['color'] == container_color]
                        if not containers: return None
                        # Use the first one or similar heuristic
                        c_blob = containers[0] 
                        coords = c_blob['coords']
                        r_min, c_min = coords.min(axis=0)
                        r_max, c_max = coords.max(axis=0)
                        hollow_rows = range(r_min, r_max)
                        hollow_cols = range(c_min + 1, c_max)
                        cnt = 0
                        for r in hollow_rows:
                            has_az = False
                            for c in hollow_cols:
                                if ti[r, c] == azure_color:
                                    has_az = True; break
                            if not has_az: cnt += 1
                        res = np.zeros((3, 3), dtype=int)
                        fill_order = [(0,0), (0,1), (0,2), (1,2), (2,2), (2,1), (2,0), (1,0), (1,1)]
                        for i in range(min(cnt, 9)):
                            pr, pc = fill_order[i]
                            res[pr, pc] = azure_color
                        results.append(res)
                    return results
    return None
