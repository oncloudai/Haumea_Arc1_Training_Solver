import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_object_recolor_by_relative_position(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        for conn in [4, 8]:
            for crit in [lambda b: b['top_left'][1], lambda b: b['top_left'][0]]:
                for rev in [False, True]:
                    mapping = {}; consistent = True; found_change = False
                    for inp, out in solver.pairs:
                        if inp.shape != out.shape: consistent = False; break
                        blobs = sorted(get_blobs(inp, bg, conn), key=crit, reverse=rev)
                        pred = inp.copy()
                        for i, b in enumerate(blobs):
                            tr, tc = b['coords'][0]; cout = out[tr, tc]
                            if i in mapping:
                                if mapping[i] != cout: consistent = False; break
                            else: mapping[i] = cout
                            pred[b['coords'][:,0], b['coords'][:,1]] = cout
                            if cout != b['color']: found_change = True
                        if not consistent or not np.array_equal(pred, out): consistent = False; break
                    
                    if consistent and found_change:
                        # RERUN and VERIFY on ALL training pairs
                        all_match = True
                        for inp, out in solver.pairs:
                            pred = inp.copy(); blobs = sorted(get_blobs(inp, bg, conn), key=crit, reverse=rev)
                            for i, b in enumerate(blobs):
                                if i in mapping: pred[b['coords'][:,0], b['coords'][:,1]] = mapping[i]
                            if not np.array_equal(pred, out): all_match = False; break
                        if not all_match: consistent = False

                    if consistent and found_change:
                        results = []
                        for ti in solver.test_in:
                            res = ti.copy(); t_blobs = sorted(get_blobs(ti, bg, conn), key=crit, reverse=rev)
                            for i, b in enumerate(t_blobs):
                                if i in mapping: res[b['coords'][:,0], b['coords'][:,1]] = mapping[i]
                            results.append(res)
                        return results
    return None
