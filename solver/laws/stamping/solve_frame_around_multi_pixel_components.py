import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_frame_around_multi_pixel_components(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        for c_in in range(10):
            if c_in == bg: continue
            for c_frame in range(10):
                if c_frame == bg or c_frame == c_in: continue
                
                for conn in [4, 8]:
                    consistent = True
                    found_any = False
                    
                    for inp, out in solver.pairs:
                        if inp.shape != out.shape: consistent = False; break
                        
                        pred = inp.copy()
                        blobs = get_blobs(inp, bg, connectivity=conn)
                        has_large = False
                        for b in blobs:
                            if b['color'] == c_in and b['size'] > 1:
                                has_large = True
                                r_min, c_min = b['coords'].min(axis=0)
                                r_max, c_max = b['coords'].max(axis=0)
                                for r in range(r_min - 1, r_max + 2):
                                    for c in range(c_min - 1, c_max + 2):
                                        if 0 <= r < pred.shape[0] and 0 <= c < pred.shape[1]:
                                            if r == r_min - 1 or r == r_max + 1 or c == c_min - 1 or c == c_max + 1:
                                                pred[r, c] = c_frame
                        
                        if not np.array_equal(pred, out):
                            consistent = False; break
                        if has_large: found_any = True
                    
                    if consistent and found_any:
                        results = []
                        for ti in solver.test_in:
                            res = ti.copy()
                            t_blobs = get_blobs(ti, bg, connectivity=conn)
                            for b in t_blobs:
                                if b['color'] == c_in and b['size'] > 1:
                                    r_min, c_min = b['coords'].min(axis=0)
                                    r_max, c_max = b['coords'].max(axis=0)
                                    for r in range(r_min - 1, r_max + 2):
                                        for c in range(c_min - 1, c_max + 2):
                                            if 0 <= r < res.shape[0] and 0 <= c < res.shape[1]:
                                                if r == r_min - 1 or r == r_max + 1 or c == c_min - 1 or c == c_max + 1:
                                                    res[r, c] = c_frame
                            results.append(res)
                        return results
    return None
