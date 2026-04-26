import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_bar_pivot_extension(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        for c_bar in range(1, 10):
            if c_bar == bg: continue
            for c_pivot in range(1, 10):
                if c_pivot == bg or c_pivot == c_bar: continue
                
                consistent = True
                found_any = False
                for inp, out in solver.pairs:
                    if inp.shape != out.shape: consistent = False; break
                    pred = inp.copy()
                    
                    mask = (inp == c_bar) | (inp == c_pivot)
                    labeled, n = label(mask, structure=np.ones((3,3)))
                    
                    for i in range(1, n+1):
                        coords = np.argwhere(labeled == i)
                        p_coords = [tuple(p) for p in coords if inp[p[0], p[1]] == c_pivot]
                        if len(p_coords) != 1: continue
                        r_p, c_p = p_coords[0]
                        
                        b_coords = [p for p in coords if inp[p[0], p[1]] == c_bar]
                        if not b_coords: continue
                        
                        r_min, c_min = coords.min(axis=0)
                        r_max, c_max = coords.max(axis=0)
                        h, w = r_max - r_min + 1, c_max - c_min + 1
                        
                        if h > w: # Vertical bar
                            W = w
                            if c_p == c_min: # Extend Left
                                rows_range = range(r_p - (W-1), r_p + W)
                                cols_range = range(0, c_max + 1)
                                for r in rows_range:
                                    for c in cols_range:
                                        if 0 <= r < pred.shape[0] and 0 <= c < pred.shape[1]:
                                            if r == r_p and c <= c_p: pred[r, c] = c_pivot
                                            else: pred[r, c] = c_bar
                                            found_any = True
                            elif c_p == c_max: # Extend Right
                                rows_range = range(r_p - (W-1), r_p + W)
                                cols_range = range(c_min, pred.shape[1])
                                for r in rows_range:
                                    for c in cols_range:
                                        if 0 <= r < pred.shape[0] and 0 <= c < pred.shape[1]:
                                            if r == r_p and c >= c_p: pred[r, c] = c_pivot
                                            else: pred[r, c] = c_bar
                                            found_any = True
                        elif w > h: # Horizontal bar
                            H = h
                            if r_p == r_min: # Extend Up
                                rows_range = range(0, r_max + 1)
                                cols_range = range(c_p - (H-1), c_p + H)
                                for r in rows_range:
                                    for c in cols_range:
                                        if 0 <= r < pred.shape[0] and 0 <= c < pred.shape[1]:
                                            if c == c_p and r <= r_p: pred[r, c] = c_pivot
                                            else: pred[r, c] = c_bar
                                            found_any = True
                            elif r_p == r_max: # Extend Down
                                rows_range = range(r_min, pred.shape[0])
                                cols_range = range(c_p - (H-1), c_p + H)
                                for r in rows_range:
                                    for c in cols_range:
                                        if 0 <= r < pred.shape[0] and 0 <= c < pred.shape[1]:
                                            if c == c_p and r >= r_p: pred[r, c] = c_pivot
                                            else: pred[r, c] = c_bar
                                            found_any = True
                    
                    if not np.array_equal(pred, out):
                        consistent = False; break
                
                if consistent and found_any:
                    def apply(grid):
                        res = grid.copy()
                        mask = (grid == c_bar) | (grid == c_pivot)
                        labeled, n = label(mask, structure=np.ones((3,3)))
                        for i in range(1, n+1):
                            coords = np.argwhere(labeled == i)
                            p_coords = [tuple(p) for p in coords if grid[p[0], p[1]] == c_pivot]
                            if len(p_coords) != 1: continue
                            r_p, c_p = p_coords[0]
                            r_min, c_min = coords.min(axis=0)
                            r_max, c_max = coords.max(axis=0)
                            h, w = r_max - r_min + 1, c_max - c_min + 1
                            if h > w:
                                W = w
                                if c_p == c_min:
                                    for r in range(r_p - (W-1), r_p + W):
                                        for c in range(0, c_max + 1):
                                            if 0 <= r < res.shape[0] and 0 <= c < res.shape[1]:
                                                if r == r_p and c <= c_p: res[r, c] = c_pivot
                                                else: res[r, c] = c_bar
                                elif c_p == c_max:
                                    for r in range(r_p - (W-1), r_p + W):
                                        for c in range(c_min, res.shape[1]):
                                            if 0 <= r < res.shape[0] and 0 <= c < res.shape[1]:
                                                if r == r_p and c >= c_p: res[r, c] = c_pivot
                                                else: res[r, c] = c_bar
                            elif w > h:
                                H = h
                                if r_p == r_min:
                                    for r in range(0, r_max + 1):
                                        for c in range(c_p - (H-1), c_p + H):
                                            if 0 <= r < res.shape[0] and 0 <= c < res.shape[1]:
                                                if c == c_p and r <= r_p: res[r, c] = c_pivot
                                                else: res[r, c] = c_bar
                                elif r_p == r_max:
                                    for r in range(r_min, res.shape[0]):
                                        for c in range(c_p - (H-1), c_p + H):
                                            if 0 <= r < res.shape[0] and 0 <= c < res.shape[1]:
                                                if c == c_p and r >= r_p: res[r, c] = c_pivot
                                                else: res[r, c] = c_bar
                        return res
                    return [apply(ti) for ti in solver.test_in]
    return None
