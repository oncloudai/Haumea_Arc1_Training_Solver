import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_targeted_gravity(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        for static_c, mobile_c in [(8, 2), (2, 8), (5, 8), (8, 5)]:
            consistent = True; found_move = False
            for inp, out in solver.pairs:
                blobs = get_blobs(inp, bg)
                sb = [b for b in blobs if b['color'] == static_c]; mb = [b for b in blobs if b['color'] == mobile_c]
                if not sb or not mb: consistent = False; break
                mr, mc = mb[0]['top_left']; sr, sc = sb[0]['top_left']; dr, dc = (np.sign(sr-mr), 0) if abs(sr-mr) > abs(sc-mc) else (0, np.sign(sc-mc))
                curr = mb[0]['coords'].copy(); static_mask = (inp == static_c)
                while True:
                    nxt = curr + (dr, dc)
                    if np.any(nxt < 0) or np.any(nxt[:,0] >= inp.shape[0]) or np.any(nxt[:,1] >= inp.shape[1]): break
                    if any(static_mask[r,c] for r,c in nxt): break
                    curr = nxt; found_move = True
                pred = inp.copy(); pred[mb[0]['coords'][:,0], mb[0]['coords'][:,1]] = bg; pred[curr[:,0], curr[:,1]] = mobile_c
                if not np.array_equal(pred, out): consistent = False; break
            if consistent and found_move:
                results = []
                for ti in solver.test_in:
                    res = ti.copy(); blobs = get_blobs(ti, bg); sb = [b for b in blobs if b['color'] == static_c]; mb = [b for b in blobs if b['color'] == mobile_c]
                    if not sb or not mb: results.append(ti); continue
                    mr, mc = mb[0]['top_left']; sr, sc = sb[0]['top_left']; dr, dc = (np.sign(sr-mr), 0) if abs(sr-mr) > abs(sc-mc) else (0, np.sign(sc-mc))
                    curr = mb[0]['coords'].copy(); sm = (ti == static_c)
                    while True:
                        nxt = curr + (dr, dc)
                        if np.any(nxt < 0) or np.any(nxt[:,0] >= ti.shape[0]) or np.any(nxt[:,1] >= ti.shape[1]): break
                        if any(sm[r,c] for r,c in nxt): break
                        curr = nxt
                    res[mb[0]['coords'][:,0], mb[0]['coords'][:,1]] = bg; res[curr[:,0], curr[:,1]] = mobile_c; results.append(res)
                return results
    return None
