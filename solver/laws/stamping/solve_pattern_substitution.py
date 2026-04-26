import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_pattern_substitution(solver) -> Optional[List[np.ndarray]]:
    bg = 0
    for conn in [4, 8]:
        if not solver.train_in: continue
        blobs0 = get_blobs(solver.train_in[0], bg, conn)
        if len(blobs0) < 2: continue
        for tidx in range(len(blobs0)):
            target_color = blobs0[tidx]['color']
            all_match = True
            for inp, out in solver.pairs:
                blobs = get_blobs(inp, bg, conn); curr_tmpl = [b for b in blobs if b['color'] == target_color]
                if not curr_tmpl: all_match = False; break
                tmpl = curr_tmpl[0]; targets = [b for b in blobs if b is not tmpl]
                tr1, tc1 = tmpl['coords'].min(axis=0); tr2, tc2 = tmpl['coords'].max(axis=0); th, tw = tr2-tr1+1, tc2-tc1+1; t_rel = tmpl['coords'] - [tr1, tc1]; t_mid_r, t_mid_c = (tr1+tr2)/2.0, (tc1+tc2)/2.0
                pred = np.zeros_like(out); pred[tmpl['coords'][:,0], tmpl['coords'][:,1]] = tmpl['color']
                for t in targets:
                    trr1, tcc1 = t['coords'].min(axis=0); trr2, tcc2 = t['coords'].max(axis=0); tm_r, tm_c = (trr1+trr2)/2.0, (tcc1+tcc2)/2.0; dirs = []
                    if abs(tm_r - t_mid_r) < 0.1: dirs.extend([(0, 1), (0, -1)])
                    if abs(tm_c - t_mid_c) < 0.1: dirs.extend([(1, 0), (-1, 0)])
                    for dr, dc in dirs:
                        for m in range(15):
                            nr, nc = int(round(trr1 + m * dr * (th+1))), int(round(tcc1 + m * dc * (tw+1)))
                            if nr < 0 or nr >= out.shape[0] or nc < 0 or nc >= out.shape[1]: break
                            for pr, pc in t_rel:
                                if 0 <= nr+pr < pred.shape[0] and 0 <= nc+pc < pred.shape[1]: pred[nr+pr, nc+pc] = t['color']
                pred[tmpl['coords'][:,0], tmpl['coords'][:,1]] = tmpl['color']
                if not np.array_equal(pred, out): all_match = False; break
            if all_match:
                results = []
                for ti in solver.test_in:
                    blobs = get_blobs(ti, bg, conn); curr_tmpl = [b for b in blobs if b['color'] == target_color]
                    if not curr_tmpl: return None
                    tmpl = curr_tmpl[0]; targets = [b for b in blobs if b is not tmpl]
                    res = np.zeros_like(ti); res[tmpl['coords'][:,0], tmpl['coords'][:,1]] = tmpl['color']
                    tr1, tc1 = tmpl['coords'].min(axis=0); tr2, tc2 = tmpl['coords'].max(axis=0); th, tw = tr2-tr1+1, tc2-tc1+1; t_rel = tmpl['coords'] - [tr1, tc1]; t_mid_r, t_mid_c = (tr1+tr2)/2.0, (tc1+tc2)/2.0
                    for t in targets:
                        trr1, tcc1 = t['coords'].min(axis=0); trr2, tcc2 = t['coords'].max(axis=0); tm_r, tm_c = (trr1+trr2)/2.0, (tcc1+tcc2)/2.0; dirs = []
                        if abs(tm_r - t_mid_r) < 0.1: dirs.extend([(0, 1), (0, -1)])
                        if abs(tm_c - t_mid_c) < 0.1: dirs.extend([(1, 0), (-1, 0)])
                        for dr, dc in dirs:
                            for m in range(15):
                                nr, nc = int(round(trr1 + m * dr * (th+1))), int(round(tcc1 + m * dc * (tw+1)))
                                if nr < 0 or nr >= ti.shape[0] or nc < 0 or nc >= ti.shape[1]: break
                                for pr, pc in t_rel:
                                    if 0 <= nr+pr < res.shape[0] and 0 <= nc+pc < res.shape[1]: res[nr+pr, nc+pc] = t['color']
                    res[tmpl['coords'][:,0], tmpl['coords'][:,1]] = tmpl['color']; results.append(res)
                return results
    return None
