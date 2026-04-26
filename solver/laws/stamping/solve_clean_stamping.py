import numpy as np
from typing import List, Optional

def solve_clean_stamping(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        stamps = {}; consistent = True; found_any = False
        for inp, out in solver.pairs:
            if inp.shape != out.shape: consistent = False; break
            added = (out != inp) & (out != bg); added_coords = np.argwhere(added); in_coords = np.argwhere(inp != bg)
            if len(in_coords) == 0:
                if len(added_coords) > 0: consistent = False; break
                continue
            for ar, ac in added_coords:
                dists = np.abs(in_coords[:, 0] - ar) + np.abs(in_coords[:, 1] - ac)
                ir, ic = in_coords[np.argmin(dists)]; cin, cout = inp[ir, ic], out[ar, ac]
                if cin not in stamps: stamps[cin] = set()
                stamps[cin].add((ar - ir, ac - ic, cout)); found_any = True
        if consistent and found_any:
            for inp, out in solver.pairs:
                pred = inp.copy()
                for r, c in np.argwhere(inp != bg):
                    cin = inp[r, c]
                    if cin in stamps:
                        for dr, dc, cout in stamps[cin]:
                            if 0 <= r+dr < pred.shape[0] and 0 <= c+dc < pred.shape[1]: pred[r+dr, c+dc] = cout
                if not np.array_equal(pred, out): consistent = False; break
            if consistent:
                results = []
                for ti in solver.test_in:
                    res = ti.copy()
                    for r, c in np.argwhere(ti != bg):
                        if ti[r, c] in stamps:
                            for dr, dc, cout in stamps[ti[r, c]]:
                                if 0 <= r+dr < res.shape[0] and 0 <= c+dc < res.shape[1]: res[r+dr, c+dc] = cout
                    results.append(res)
                return results
    return None
