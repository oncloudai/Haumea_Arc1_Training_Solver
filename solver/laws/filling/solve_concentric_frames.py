
import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_concentric_frames(solver) -> Optional[List[np.ndarray]]:
    def process(grid):
        h, w = grid.shape
        out = np.zeros_like(grid)
        labeled, n = label(grid != 0)
        for i in range(1, n + 1):
            coords = np.argwhere(labeled == i)
            r1, r2 = coords[:,0].min(), coords[:,0].max()
            c1, c2 = coords[:,1].min(), coords[:,1].max()
            obj_rows = set(coords[:, 0])
            obj_cols = set(coords[:, 1])
            
            obj = grid[r1:r2+1, c1:c2+1]
            c1_color = obj[0, 0]
            inner_mask = (obj != c1_color) & (obj != 0)
            inner_coords = np.argwhere(inner_mask)
            if len(inner_coords) == 0: continue
            
            ir1, jc1 = inner_coords.min(axis=0); ir2, jc2 = inner_coords.max(axis=0)
            inner_h, inner_w = ir2 - ir1 + 1, jc2 - jc1 + 1
            c0_color = obj[ir1, jc1]
            
            gr1, gc1 = r1 + ir1, c1 + jc1
            gr2, gc2 = r1 + ir2, c1 + jc2
            
            K = 1 + inner_h
            for k in range(K, -1, -1):
                color = c1_color if (k != 1) else c0_color
                cr1, cc1 = gr1 - k, gc1 - k
                cr2, cc2 = gr2 + k, gc2 + k
                for r in range(cr1, cr2 + 1):
                    for c in range(cc1, cc2 + 1):
                        if r == cr1 or r == cr2 or c == cc1 or c == cc2 or k == 0:
                            if r in obj_rows or c in obj_cols:
                                if 0 <= r < h and 0 <= c < w:
                                    out[r, c] = color
        return out

    for inp, out in solver.pairs:
        if not np.array_equal(process(inp), out): return None
    return [process(ti) for ti in solver.test_in]
