import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_recolor_to_frame(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        for c_in in range(1, 10):
            if c_in == bg: continue
            # Common frame colors
            for c_corner, c_edge, c_interior in [(1, 4, 2)]:
                consistent = True
                found_any = False
                for inp, out in solver.pairs:
                    if inp.shape != out.shape: consistent = False; break
                    pred = inp.copy()
                    blobs = get_blobs(inp, bg, connectivity=8)
                    for b in blobs:
                        if b['color'] == c_in:
                            r1, c1 = b['coords'].min(axis=0)
                            r2, c2 = b['coords'].max(axis=0)
                            h, w = r2 - r1 + 1, c2 - c1 + 1
                            if h >= 2 and w >= 2:
                                for r in range(r1, r2 + 1):
                                    for c in range(c1, c2 + 1):
                                        is_corner = (r == r1 or r == r2) and (c == c1 or c == c2)
                                        is_edge = (r == r1 or r == r2) or (c == c1 or c == c2)
                                        if is_corner: pred[r, c] = c_corner
                                        elif is_edge: pred[r, c] = c_edge
                                        else: pred[r, c] = c_interior
                                found_any = True
                    if not np.array_equal(pred, out):
                        consistent = False; break
                
                if consistent and found_any:
                    def apply(grid):
                        res = grid.copy()
                        blobs = get_blobs(grid, bg, connectivity=8)
                        for b in blobs:
                            if b['color'] == c_in:
                                r1, c1 = b['coords'].min(axis=0)
                                r2, c2 = b['coords'].max(axis=0)
                                h, w = r2 - r1 + 1, c2 - c1 + 1
                                if h >= 2 and w >= 2:
                                    for r in range(r1, r2+1):
                                        for c in range(c1, c2+1):
                                            is_corner = (r == r1 or r == r2) and (c == c1 or c == c2)
                                            is_edge = (r == r1 or r == r2) or (c == c1 or c == c2)
                                            if is_corner: res[r, c] = c_corner
                                            elif is_edge: res[r, c] = c_edge
                                            else: res[r, c] = c_interior
                        return res
                    return [apply(ti) for ti in solver.test_in]
    return None
