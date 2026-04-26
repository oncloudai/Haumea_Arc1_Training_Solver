import numpy as np
from typing import List, Optional

def solve_one(inp):
    inp = np.array(inp)
    h, w = inp.shape
    red_coords = np.argwhere(inp == 2)
    green_coords = np.argwhere(inp == 3)
    if len(red_coords) == 0 or len(green_coords) == 0: return inp
    
    r_min, r_max = green_coords[:, 0].min(), green_coords[:, 0].max()
    c_min, c_max = green_coords[:, 1].min(), green_coords[:, 1].max()
    is_horizontal = (r_max == r_min)
    
    tr_min, tr_max = red_coords[:, 0].min(), red_coords[:, 0].max()
    tc_min, tc_max = red_coords[:, 1].min(), red_coords[:, 1].max()
    
    out = inp.copy()
    target_rows = list(range(tr_min, tr_max + 1))
    target_cols = list(range(tc_min, tc_max + 1))
    
    if not is_horizontal: # Vertical marker (1xN) - moves vertically then horizontally then vertically
        dr = -1 if tr_min < r_min else (1 if tr_min > r_min else 0)
        if dr == 0: dr = -1
        
        head = green_coords[np.argmin(green_coords[:, 0])] if dr == -1 else green_coords[np.argmax(green_coords[:, 0])]
        
        best_r = head[0]
        best_tc = -1
        curr_r = head[0]
        
        # Move vertically as far as possible
        while 0 <= curr_r + dr < h and inp[curr_r + dr, head[1]] != 8:
            curr_r += dr
            # Check if we can turn horizontally and then vertically to reach target
            for tc in target_cols:
                h_clear = True
                c1, c2 = min(head[1], tc), max(head[1], tc)
                for c in range(c1, c2 + 1):
                    if inp[curr_r, c] == 8:
                        h_clear = False; break
                if h_clear:
                    v_clear = True
                    r1, r2 = min(curr_r, tr_min), max(curr_r, tr_max)
                    for r in range(r1, r2 + 1):
                        if inp[r, tc] == 8:
                            v_clear = False; break
                    if v_clear:
                        best_r = curr_r
                        best_tc = tc
        
        if best_tc == -1: best_tc = tc_min
            
        # Segment 1: Vertical
        r1, r2 = min(head[0], best_r), max(head[0], best_r)
        for r in range(r1, r2 + 1):
            if out[r, head[1]] == 0: out[r, head[1]] = 3
        # Segment 2: Horizontal
        c1, c2 = min(head[1], best_tc), max(head[1], best_tc)
        for c in range(c1, c2 + 1):
            if out[best_r, c] == 0: out[best_r, c] = 3
        # Segment 3: Vertical to target
        tr_target_edge = tr_min - 1 if best_r < tr_min else tr_max + 1
        tr_target_edge = max(0, min(h - 1, tr_target_edge))
        r1, r2 = min(best_r, tr_target_edge), max(best_r, tr_target_edge)
        for r in range(r1, r2 + 1):
            if out[r, best_tc] == 0: out[r, best_tc] = 3
        
    else: # Horizontal marker (Nx1) - moves horizontally then vertically then horizontally
        dc = 1 if tc_min > c_max else (-1 if tc_max < c_min else 0)
        if dc == 0: dc = 1
            
        head = green_coords[np.argmax(green_coords[:, 1])] if dc == 1 else green_coords[np.argmin(green_coords[:, 1])]
        
        best_c = head[1]
        best_tr = -1
        curr_c = head[1]
        
        while 0 <= curr_c + dc < w and inp[head[0], curr_c + dc] != 8:
            curr_c += dc
            for tr in target_rows:
                v_clear = True
                r1, r2 = min(head[0], tr), max(head[0], tr)
                for r in range(r1, r2 + 1):
                    if inp[r, curr_c] == 8:
                        v_clear = False; break
                if v_clear:
                    h_clear = True
                    c1, c2 = min(curr_c, tc_min), max(curr_c, tc_max)
                    for c in range(c1, c2 + 1):
                        if inp[tr, c] == 8:
                            h_clear = False; break
                    if h_clear:
                        best_c = curr_c
                        best_tr = tr
        
        if best_tr == -1: best_tr = tr_min
        
        c1, c2 = min(head[1], best_c), max(head[1], best_c)
        for c in range(c1, c2 + 1):
            if out[head[0], c] == 0: out[head[0], c] = 3
        r1, r2 = min(head[0], best_tr), max(head[0], best_tr)
        for r in range(r1, r2 + 1):
            if out[r, best_c] == 0: out[r, best_c] = 3
        tc_target_edge = tc_min - 1 if best_c < tc_min else tc_max + 1
        tc_target_edge = max(0, min(w - 1, tc_target_edge))
        c1, c2 = min(best_c, tc_target_edge), max(best_c, tc_target_edge)
        for c in range(c1, c2 + 1):
            if out[best_tr, c] == 0: out[best_tr, c] = 3
            
    return out

def solve_steered_pathfinding(solver) -> Optional[List[np.ndarray]]:
    """
    Crawlers start from green marker and reach red target by an L-shaped or Z-shaped path (up to 3 segments).
    Obstacles (color 8) are avoided.
    """
    for i, (inp, out) in enumerate(solver.pairs):
        pred = solve_one(inp)
        if not np.array_equal(pred, out):
            return None
    return [solve_one(ti) for ti in solver.test_in]
