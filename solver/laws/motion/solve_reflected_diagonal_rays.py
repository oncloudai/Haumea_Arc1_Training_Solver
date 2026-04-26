import numpy as np
from typing import List, Optional
from collections import Counter

def solve_reflected_diagonal_rays(solver) -> Optional[List[np.ndarray]]:
    consistent = True; found_any = False
    for inp, out in solver.pairs:
        h, w = inp.shape
        counts = Counter(inp.flatten())
        # Seeds are colors that appear once
        seeds = [c for c, count in counts.items() if count == 1]
        if not seeds: consistent = False; break
        
        pred = inp.copy()
        for s_color in seeds:
            r0, c0 = np.argwhere(inp == s_color)[0]
            # Find bg color
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < h and 0 <= nc < w: neighbors.append(inp[nr, nc])
            if not neighbors: continue
            valid_bg = [n for n in neighbors if n not in seeds]
            if not valid_bg: continue
            bg_color = Counter(valid_bg).most_common(1)[0][0]
            
            # 4 diagonal rays
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cr, cc = r0, c0
                cdr, cdc = dr, dc
                # Limit steps to avoid infinite loops in small grids
                for _ in range(h * w * 2):
                    nr, nc = cr + cdr, cc + cdc
                    
                    # Reflection logic
                    hit_boundary = False
                    if nr < 0 or nr >= h:
                        cdr = -cdr; hit_boundary = True
                    if nc < 0 or nc >= w:
                        cdc = -cdc; hit_boundary = True
                    
                    if hit_boundary:
                        # Re-calculate nr, nc with new direction
                        nr, nc = cr + cdr, cc + cdc
                        # If still out of bounds (corner), reflect both
                        if nr < 0 or nr >= h or nc < 0 or nc >= w:
                            # This should be handled by the double-hit above, but just in case
                            break
                    
                    cr, cc = nr, nc
                    if inp[cr, cc] == bg_color:
                        pred[cr, cc] = s_color
                        found_any = True
                    elif cr == r0 and cc == c0:
                        # Back at start
                        break
                        
        if not np.array_equal(pred, out):
            consistent = False; break
            
    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            h, w = ti.shape; res = ti.copy()
            counts = Counter(ti.flatten())
            seeds = [c for c, count in counts.items() if count == 1]
            for s_color in seeds:
                r0, c0 = np.argwhere(ti == s_color)[0]
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r0 + dr, c0 + dc
                    if 0 <= nr < h and 0 <= nc < w: neighbors.append(ti[nr, nc])
                if not neighbors: continue
                valid_bg = [n for n in neighbors if n not in seeds]
                if not valid_bg: continue
                bg_color = Counter(valid_bg).most_common(1)[0][0]
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    cr, cc, cdr, cdc = r0, c0, dr, dc
                    for _ in range(h * w * 2):
                        nr, nc = cr + cdr, cc + cdc
                        hit = False
                        if nr < 0 or nr >= h: cdr = -cdr; hit = True
                        if nc < 0 or nc >= w: cdc = -cdc; hit = True
                        if hit: nr, nc = cr + cdr, cc + cdc
                        if nr < 0 or nr >= h or nc < 0 or nc >= w: break
                        cr, cc = nr, nc
                        if ti[cr, cc] == bg_color: res[cr, cc] = s_color
                        elif cr == r0 and cc == c0: break
            results.append(res)
        return results
    return None
