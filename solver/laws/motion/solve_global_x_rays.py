
import numpy as np
from typing import List, Optional
from collections import Counter

def solve_global_x_rays(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies seeds (rare colors) and their associated background colors.
    Propagates diagonal rays from all seeds globally.
    Pixels on rays are painted with the seed color matching their background color.
    """
    consistent = True; found_any = False
    all_pairs_bg_to_seed = {} # Global mapping if possible
    
    # First pass: identify bg_to_seed for each pair
    pair_datas = []
    for inp, out in solver.pairs:
        h, w = inp.shape
        counts = Counter(inp.flatten())
        # Backgrounds have many pixels
        bg_colors = [c for c, count in counts.items() if count > 5]
        
        seeds = []
        bg_to_seed = {}
        for r in range(h):
            for c in range(w):
                if inp[r, c] not in bg_colors:
                    s_color = inp[r, c]
                    neighbors = []
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and inp[nr, nc] in bg_colors:
                            neighbors.append(inp[nr, nc])
                    if neighbors:
                        home_bg = Counter(neighbors).most_common(1)[0][0]
                        seeds.append((r, c, s_color))
                        bg_to_seed[home_bg] = s_color
        
        pair_datas.append((bg_colors, seeds, bg_to_seed))
        
    for i, (inp, out) in enumerate(solver.pairs):
        h, w = inp.shape
        bg_colors, seeds, bg_to_seed = pair_datas[i]
        pred = inp.copy()
        
        for s_r, s_c, s_color in seeds:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cr, cc = s_r, s_c
                while True:
                    cr += dr
                    cc += dc
                    if 0 <= cr < h and 0 <= cc < w:
                        target_bg = inp[cr, cc]
                        if target_bg in bg_to_seed:
                            pred[cr, cc] = bg_to_seed[target_bg]
                            found_any = True
                    else:
                        break
        
        if not np.array_equal(pred, out):
            consistent = False; break
            
    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            h, w = ti.shape; res = ti.copy()
            counts = Counter(ti.flatten())
            bg_colors = [c for c, count in counts.items() if count > 5]
            seeds = []
            bg_to_seed = {}
            for r in range(h):
                for c in range(w):
                    if ti[r, c] not in bg_colors:
                        s_color = ti[r, c]
                        neighbors = []
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and ti[nr, nc] in bg_colors:
                                neighbors.append(ti[nr, nc])
                        if neighbors:
                            home_bg = Counter(neighbors).most_common(1)[0][0]
                            seeds.append((r, c, s_color))
                            bg_to_seed[home_bg] = s_color
            
            for s_r, s_c, s_color in seeds:
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    cr, cc = s_r, s_c
                    while True:
                        cr += dr
                        cc += dc
                        if 0 <= cr < h and 0 <= cc < w:
                            target_bg = ti[cr, cc]
                            if target_bg in bg_to_seed:
                                res[cr, cc] = bg_to_seed[target_bg]
                        else:
                            break
            results.append(res)
        return results
    return None
