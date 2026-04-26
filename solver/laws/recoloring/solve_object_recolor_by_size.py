
import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_object_recolor_by_size(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        for conn in [4, 8]:
            smap = {}; consistent = True; found_change = False
            for inp, out in solver.pairs:
                if inp.shape != out.shape: consistent = False; break
                h, w = inp.shape
                blobs = get_blobs(inp, bg, conn)
                pred = inp.copy()
                for b in blobs:
                    r, c = b['coords'][0]
                    target_color = out[r, c]
                    for pr, pc in b['coords']:
                        if out[pr, pc] != target_color: consistent = False; break
                    if not consistent: break
                    
                    if b['size'] in smap and smap[b['size']] != target_color: consistent = False; break
                    smap[b['size']] = target_color
                    for pr, pc in b['coords']: pred[pr, pc] = target_color
                    if target_color != b['color']: found_change = True
                if not consistent: break
                
                for r in range(h):
                    for c in range(w):
                        if out[r, c] != bg:
                            if pred[r, c] != out[r, c]: consistent = False; break
                    if not consistent: break
                if not consistent: break

                if not np.array_equal(pred, out): consistent = False; break
            
            if consistent and found_change:
                results = []
                for ti in solver.test_in:
                    res = ti.copy()
                    for b in get_blobs(ti, bg, conn):
                        if b['size'] in smap:
                            for pr, pc in b['coords']: res[pr, pc] = smap[b['size']]
                    results.append(res)
                return results
    return None
