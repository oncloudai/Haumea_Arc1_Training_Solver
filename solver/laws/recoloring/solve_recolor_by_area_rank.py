import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_recolor_by_area_rank(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        for conn in [4, 8]:
            # Area of bounding box
            def area(b):
                min_r, min_c = b['coords'].min(axis=0)
                max_r, max_c = b['coords'].max(axis=0)
                return (max_r - min_r + 1) * (max_c - min_c + 1)
            
            strategy = lambda b: (-area(b), b['top_left'][0], b['top_left'][1])
            mapping = {}; consistent = True; found_change = False
            for inp, out in solver.pairs:
                if inp.shape != out.shape: consistent = False; break
                blobs = sorted(get_blobs(inp, bg, conn), key=strategy)
                pred = inp.copy()
                for i, b in enumerate(blobs):
                    r, c = b['coords'][0]; target_color = out[r, c]
                    if i in mapping and mapping[i] != target_color: consistent = False; break
                    mapping[i] = target_color
                    for pr, pc in b['coords']: pred[pr, pc] = target_color
                    if target_color != b['color']: found_change = True
                if not consistent or not np.array_equal(pred, out): consistent = False; break
            
            if consistent and found_change:
                results = []
                for ti in solver.test_in:
                    res = ti.copy(); blobs = sorted(get_blobs(ti, bg, conn), key=strategy)
                    for i, b in enumerate(blobs):
                        if i in mapping:
                            for pr, pc in b['coords']: res[pr, pc] = mapping[i]
                    results.append(res)
                return results
    return None
