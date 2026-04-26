import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_object_template_replace(solver) -> Optional[List[np.ndarray]]:
    bg = 0
    for target_c in range(1, 10):
        for conn in [4, 8]:
            consistent = True; found_any = False
            for inp, out in solver.pairs:
                blobs = get_blobs(inp, bg, conn)
                targets = [b for b in blobs if np.all(inp[b['coords'][:,0], b['coords'][:,1]] == target_c)]
                templates = [b for b in blobs if not np.all(inp[b['coords'][:,0], b['coords'][:,1]] == target_c)]
                if not targets or len(templates) != 1: consistent = False; break
                tmpl = templates[0]; pred = inp.copy()
                for r, c in tmpl['coords']: pred[r, c] = bg
                for t in targets:
                    if t['normalized'] != tmpl['normalized']: consistent = False; break
                    dr, dc = t['top_left'] - tmpl['top_left']
                    for r, c in tmpl['coords']:
                        if 0 <= r+dr < pred.shape[0] and 0 <= c+dc < pred.shape[1]:
                            pred[r+dr, c+dc] = inp[r, c]
                if not consistent or not np.array_equal(pred, out): consistent = False; break
                found_any = True
            if consistent and found_any:
                results = []
                for ti in solver.test_in:
                    blobs = get_blobs(ti, bg, conn)
                    targets = [b for b in blobs if np.all(ti[b['coords'][:,0], b['coords'][:,1]] == target_c)]
                    tmpl_candidates = [b for b in blobs if not np.all(ti[b['coords'][:,0], b['coords'][:,1]] == target_c)]
                    if not tmpl_candidates: return None
                    tmpl = tmpl_candidates[0]; res = ti.copy()
                    for r, c in tmpl['coords']: res[r, c] = bg
                    for t in targets:
                        dr, dc = t['top_left'] - tmpl['top_left']
                        for r, c in tmpl['coords']:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < res.shape[0] and 0 <= nc < res.shape[1]: res[nr, nc] = ti[r, c]
                    results.append(res)
                return results
    return None
