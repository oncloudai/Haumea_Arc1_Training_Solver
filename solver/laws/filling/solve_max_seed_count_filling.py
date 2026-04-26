import numpy as np
from typing import List, Optional

def solve_max_seed_count_filling(solver) -> Optional[List[np.ndarray]]:
    delim = 5; bg = 0
    consistent = True; found_any = False
    for inp, out in solver.pairs:
        rows = np.where(np.all(inp == delim, axis=1))[0]; cols = np.where(np.all(inp == delim, axis=0))[0]
        if len(rows) == 0 and len(cols) == 0: consistent = False; break
        r_bounds = [-1] + sorted(list(rows)) + [inp.shape[0]]
        c_bounds = [-1] + sorted(list(cols)) + [inp.shape[1]]
        counts = []; blocks = []
        for r_idx in range(len(r_bounds)-1):
            for c_idx in range(len(c_bounds)-1):
                r1, r2 = r_bounds[r_idx]+1, r_bounds[r_idx+1]; c1, c2 = c_bounds[c_idx]+1, c_bounds[c_idx+1]
                if r2 > r1 and c2 > c1:
                    sub = inp[r1:r2, c1:c2]; unq = np.unique(sub); seed_c = unq[(unq != bg) & (unq != delim)]
                    if len(seed_c) > 0:
                        cnt = np.sum(sub == seed_c[0]); counts.append(cnt)
                        blocks.append({'r1':r1, 'r2':r2, 'c1':c1, 'c2':c2, 'color':seed_c[0], 'count':cnt})
                    else: blocks.append({'r1':r1, 'r2':r2, 'c1':c1, 'c2':c2, 'color':bg, 'count':0}); counts.append(0)
        if not counts: consistent = False; break
        max_cnt = max(counts); pred = inp.copy()
        for b in blocks:
            if b['count'] == max_cnt: pred[b['r1']:b['r2'], b['c1']:b['c2']] = b['color']
            else: pred[b['r1']:b['r2'], b['c1']:b['c2']] = bg
        if not np.array_equal(pred, out): consistent = False; break
        found_any = True
    
    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            rows = np.where(np.all(ti == delim, axis=1))[0]; cols = np.where(np.all(ti == delim, axis=0))[0]
            r_bounds = [-1] + sorted(list(rows)) + [ti.shape[0]]; c_bounds = [-1] + sorted(list(cols)) + [ti.shape[1]]; counts = []; blocks = []
            for r_idx in range(len(r_bounds)-1):
                for c_idx in range(len(c_bounds)-1):
                    r1, r2 = r_bounds[r_idx]+1, r_bounds[r_idx+1]; c1, c2 = c_bounds[c_idx]+1, c_bounds[c_idx+1]
                    if r2 > r1 and c2 > c1:
                        sub = ti[r1:r2, c1:c2]; unq = np.unique(sub); seed_c = unq[(unq != bg) & (unq != delim)]
                        if len(seed_c) > 0:
                            cnt = np.sum(sub == seed_c[0]); counts.append(cnt)
                            blocks.append({'r1':r1, 'r2':r2, 'c1':c1, 'c2':c2, 'color':seed_c[0], 'count':cnt})
                        else: blocks.append({'r1':r1, 'r2':r2, 'c1':c1, 'c2':c2, 'color':bg, 'count':0}); counts.append(0)
            if not counts: return None
            max_cnt = max(counts); res = ti.copy()
            for b in blocks:
                if b['count'] == max_cnt: res[b['r1']:b['r2'], b['c1']:b['c2']] = b['color']
                else: res[b['r1']:b['r2'], b['c1']:b['c2']] = bg
            results.append(res)
        return results
    return None
