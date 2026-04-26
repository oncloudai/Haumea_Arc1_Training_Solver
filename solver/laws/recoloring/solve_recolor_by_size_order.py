import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_recolor_by_size_order(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        for conn in [8, 4]:
            mapping = {}; consistent = True; found_change = False
            for inp, out in solver.pairs:
                if inp.shape != out.shape: consistent = False; break
                blobs = get_blobs(inp, bg, conn)
                if not blobs: consistent = False; break
                
                # Get unique sizes and sort them
                unique_sizes = sorted(list(set([b['size'] for b in blobs])))
                
                # Check consistency of mapping size_rank -> color
                for b in blobs:
                    size_rank = unique_sizes.index(b['size'])
                    r, c = b['coords'][0]
                    target_color = out[r, c]
                    if size_rank in mapping and mapping[size_rank] != target_color:
                        consistent = False; break
                    mapping[size_rank] = target_color
                    if target_color != b['color']: found_change = True
                
                if not consistent: break
                
                # Verify that this mapping actually produces 'out'
                pred = inp.copy()
                for b in blobs:
                    size_rank = unique_sizes.index(b['size'])
                    if size_rank in mapping:
                        for pr, pc in b['coords']: pred[pr, pc] = mapping[size_rank]
                if not np.array_equal(pred, out):
                    consistent = False; break
                    
            if consistent and found_change:
                results = []
                for ti in solver.test_in:
                    res = ti.copy()
                    t_blobs = get_blobs(ti, bg, conn)
                    if not t_blobs:
                        results.append(res); continue
                    t_sizes = sorted(list(set([b['size'] for b in t_blobs])))
                    for b in t_blobs:
                        sr = t_sizes.index(b['size'])
                        if sr in mapping:
                            for pr, pc in b['coords']: res[pr, pc] = mapping[sr]
                    results.append(res)
                return results
    return None
