import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_object_wise_kronecker(solver) -> Optional[List[np.ndarray]]:
    bg = 0
    for conn in [4, 8]:
        consistent = True; found_any = False; best_stride = None
        for inp, out in solver.pairs:
            blobs = get_blobs(inp, bg, conn)
            if len(blobs) < 2: consistent = False; break
            multi = [b for b in blobs if len(b['coords']) > 1]
            if not multi: consistent = False; break
            
            worked = False
            for tmpl in multi:
                th, tw = tmpl['coords'].max(axis=0) - tmpl['coords'].min(axis=0) + 1
                for sh, sw in [(th, tw), (th+1, tw+1), (1, 1)]:
                    seeds = [b for b in blobs if b is not tmpl]
                    pred = inp.copy()
                    for s in seeds:
                        for r, c in s['coords']: pred[r, c] = bg
                    possible = True
                    for s in seeds:
                        for sr, sc in s['coords']:
                            dr, dc = sr - s['top_left'][0], sc - s['top_left'][1]
                            target_tl = s['top_left'] + (dr * sh, dc * sw)
                            for tr, tc in tmpl['coords'] - tmpl['top_left']:
                                nr, nc = target_tl[0] + tr, target_tl[1] + tc
                                if 0 <= nr < pred.shape[0] and 0 <= nc < pred.shape[1]:
                                    pred[nr, nc] = s['color']; found_any = True
                                else: possible = False; break
                            if not possible: break
                        if not possible: break
                    if possible and np.array_equal(pred, out):
                        best_stride = (sh, sw); worked = True; break
                if worked: break
            if not worked: consistent = False; break
        if consistent and found_any:
            results = []
            for ti in solver.test_in:
                t_blobs = get_blobs(ti, bg, conn)
                multi = [b for b in t_blobs if len(b['coords']) > 1]
                if not multi: return None
                tmpl = multi[0]; sh, sw = best_stride; res = ti.copy()
                seeds = [b for b in t_blobs if b is not tmpl]
                for s in seeds:
                    for r, c in s['coords']: res[r, c] = bg
                for s in seeds:
                    for sr, sc in s['coords']:
                        dr, dc = sr - s['top_left'][0], sc - s['top_left'][1]; target_tl = s['top_left'] + (dr * sh, dc * sw)
                        for tr, tc in tmpl['coords'] - tmpl['top_left']:
                            nr, nc = target_tl[0] + tr, target_tl[1] + tc
                            if 0 <= nr < res.shape[0] and 0 <= nc < res.shape[1]: res[nr, nc] = s['color']
                results.append(res)
            return results
    return None
