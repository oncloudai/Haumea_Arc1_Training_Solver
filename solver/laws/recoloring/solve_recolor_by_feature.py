import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_recolor_by_feature(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        for conn in [4, 8]:
            for feat_name in ["size", "min_r", "min_c"]:
                recolor_map = {}; consistent = True; found_any_recolor = False
                for inp, out in solver.pairs:
                    if inp.shape != out.shape: consistent = False; break
                    in_blobs = get_blobs(inp, background=bg, connectivity=conn)
                    out_blobs = get_blobs(out, background=bg, connectivity=conn)
                    if len(in_blobs) != len(out_blobs): consistent = False; break
                    for b_in in in_blobs:
                        match = next((b_out for b_out in out_blobs if np.array_equal(b_in['coords'], b_out['coords'])), None)
                        if match is None: consistent = False; break
                        out_vals = out[match['coords'][:,0], match['coords'][:,1]]
                        if len(np.unique(out_vals)) != 1: consistent = False; break
                        f_val = b_in['size'] if feat_name == "size" else b_in['top_left'][0] if feat_name == "min_r" else b_in['top_left'][1]
                        key = (f_val, b_in['color'])
                        if key in recolor_map and recolor_map[key] != match['color']: consistent = False; break
                        recolor_map[key] = match['color']
                        if match['color'] != b_in['color']: found_any_recolor = True
                    if not consistent: break
                if consistent and found_any_recolor:
                    # RERUN and VERIFY on ALL training pairs
                    all_match = True
                    for inp, out in solver.pairs:
                        pred = inp.copy(); blobs = get_blobs(inp, background=bg, connectivity=conn)
                        for b in blobs:
                            f_val = b['size'] if feat_name == "size" else b['top_left'][0] if feat_name == "min_r" else b['top_left'][1]
                            if (f_val, b['color']) in recolor_map: pred[b['coords'][:,0], b['coords'][:,1]] = recolor_map[(f_val, b['color'])]
                        if not np.array_equal(pred, out): all_match = False; break
                    if not all_match: consistent = False
                
                if consistent and found_any_recolor:
                    for ti in solver.test_in:
                        res = ti.copy(); blobs = get_blobs(ti, background=bg, connectivity=conn)
                        for b in blobs:
                            f_val = b['size'] if feat_name == "size" else b['top_left'][0] if feat_name == "min_r" else b['top_left'][1]
                            if (f_val, b['color']) in recolor_map: res[b['coords'][:,0], b['coords'][:,1]] = recolor_map[(f_val, b['color'])]
                        results.append(res)
                    return results
    return None
