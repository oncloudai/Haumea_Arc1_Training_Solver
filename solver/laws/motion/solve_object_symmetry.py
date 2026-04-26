import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_object_symmetry(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        for conn in [4, 8]:
            for op in [np.flipud, np.fliplr, np.rot90, lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 3), lambda x: np.transpose(x)]:
                consistent = True; found_any_change = False
                for inp, out in solver.pairs:
                    res = np.full_like(inp, bg); in_blobs = get_blobs(inp, bg, conn)
                    for b in in_blobs:
                        br1, bc1 = b['top_left']; br2, bc2 = b['coords'].max(axis=0); blob_grid = inp[br1:br2+1, bc1:bc2+1]
                        blob_mask = np.zeros_like(blob_grid, dtype=bool)
                        for r, c in b['coords']: blob_mask[r-br1, c-bc1] = True
                        t_blob, t_mask = op(blob_grid), op(blob_mask); th, tw = t_blob.shape
                        if br1+th > res.shape[0] or bc1+tw > res.shape[1]: consistent = False; break
                        for r in range(th):
                            for c in range(tw):
                                if t_mask[r, c]: res[br1+r, bc1+c] = t_blob[r, c]; found_any_change = True
                    if not consistent or not np.array_equal(res, out): consistent = False; break
                if consistent and found_any_change:
                    results = []
                    for ti in solver.test_in:
                        res = np.full_like(ti, bg); test_blobs = get_blobs(ti, bg, conn)
                        for b in test_blobs:
                            br1, bc1 = b['top_left']; br2, bc2 = b['coords'].max(axis=0); blob_grid = ti[br1:br2+1, bc1:bc2+1]
                            blob_mask = np.zeros_like(blob_grid, dtype=bool)
                            for r, c in b['coords']: blob_mask[r-br1, c-bc1] = True
                            t_blob, t_mask = op(blob_grid), op(blob_mask); th, tw = t_blob.shape
                            for r in range(th):
                                for c in range(tw):
                                    if t_mask[r, c] and br1+r < res.shape[0] and bc1+c < res.shape[1]: res[br1+r, bc1+c] = t_blob[r, c]
                        results.append(res)
                    return results
    return None
