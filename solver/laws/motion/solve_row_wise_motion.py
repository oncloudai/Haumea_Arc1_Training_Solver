import numpy as np
from typing import List, Optional

def solve_row_wise_motion(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        consistent = True; found_move = False; all_deltas = []
        for inp, out in solver.pairs:
            if inp.shape != out.shape: consistent = False; break
            deltas = {}
            for r in range(inp.shape[0]):
                row_in, row_out = inp[r, :], out[r, :]
                if np.array_equal(row_in, row_out): deltas[r] = 0; continue
                best_dc = None
                for dc in range(-inp.shape[1], inp.shape[1]):
                    if dc == 0: continue
                    slided = np.full_like(row_in, bg)
                    if dc > 0: slided[dc:] = row_in[:-dc]
                    else: slided[:dc] = row_in[-dc:]
                    if np.array_equal(slided, row_out): best_dc = dc; break
                if best_dc is None: consistent = False; break
                deltas[r] = best_dc; found_move = True if best_dc != 0 else found_move
            if not consistent: break
            all_deltas.append(deltas)
        if consistent and found_move:
            results = []
            for ti in solver.test_in:
                res = np.full_like(ti, bg)
                for r in range(ti.shape[0]):
                    dc = all_deltas[0].get(r, 0); row_in = ti[r, :]
                    slided = np.full_like(row_in, bg)
                    if dc > 0:
                        if dc < ti.shape[1]: slided[dc:] = row_in[:-dc]
                    elif dc < 0:
                        if -dc < ti.shape[1]: slided[:dc] = row_in[-dc:]
                    else: slided = row_in
                    res[r, :] = slided
                results.append(res)
            return results
    return None
