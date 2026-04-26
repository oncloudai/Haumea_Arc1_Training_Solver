import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_extract_smallest_bbox_object(solver) -> Optional[List[np.ndarray]]:
    consistent = True; found_any = False
    for idx, (inp, out) in enumerate(solver.pairs):
        vals, counts = np.unique(inp, return_counts=True)
        bg = vals[np.argmax(counts)]
        blobs = get_blobs(inp, bg, 8)
        if not blobs: consistent = False; break
        blob_infos = []
        for b in blobs:
            r_min, c_min = b['coords'].min(axis=0); r_max, c_max = b['coords'].max(axis=0)
            sh, sw = r_max - r_min + 1, c_max - c_min + 1
            area = sh * sw
            if area < 10: continue
            sub = inp[r_min:r_max+1, c_min:c_max+1]
            blob_infos.append({'area': area, 'c_min': c_min, 'sub': sub})
        if not blob_infos: consistent = False; break
        min_area = min(b['area'] for b in blob_infos)
        candidates = [b for b in blob_infos if b['area'] == min_area]
        candidates.sort(key=lambda x: x['c_min'])
        best = candidates[0]['sub']
        if not np.array_equal(best, out):
            # print(f"Pair {idx} fail: best shape {best.shape}, out shape {out.shape}")
            consistent = False; break
        found_any = True
    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            vals, counts = np.unique(ti, return_counts=True)
            bg = vals[np.argmax(counts)]
            blobs = get_blobs(ti, bg, 8)
            if not blobs: results.append(ti.copy()); continue
            infos = []
            for b in blobs:
                r_min, c_min = b['coords'].min(axis=0); r_max, c_max = b['coords'].max(axis=0)
                sh, sw = r_max - r_min + 1, c_max - c_min + 1
                area = sh * sw
                if area < 10: continue
                sub = ti[r_min:r_max+1, c_min:c_max+1]
                infos.append({'area': area, 'c_min': c_min, 'sub': sub})
            if not infos: results.append(ti.copy()); continue
            min_a = min(i['area'] for i in infos)
            cands = [i for i in infos if i['area'] == min_a]
            cands.sort(key=lambda x: x['c_min'])
            results.append(cands[0]['sub'])
        return results
    return None
