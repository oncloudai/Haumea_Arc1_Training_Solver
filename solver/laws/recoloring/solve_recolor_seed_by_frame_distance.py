import numpy as np
from typing import List, Optional

def solve_recolor_seed_by_frame_distance(solver) -> Optional[List[np.ndarray]]:
    def solve_single(input_grid):
        input_grid = np.array(input_grid)
        unique_colors = np.unique(input_grid)
        if 8 not in unique_colors:
            return None
        
        seed_color = 8
        frame_colors = [c for c in unique_colors if c != 0 and c != seed_color]
        
        seed_coords = np.argwhere(input_grid == seed_color)
        if len(seed_coords) == 0: return None
        s_min_r, s_min_c = seed_coords.min(axis=0)
        s_max_r, s_max_c = seed_coords.max(axis=0)
        
        frame_coords = np.argwhere(np.isin(input_grid, frame_colors))
        if len(frame_coords) == 0:
            return None
        f_min_r, f_min_c = frame_coords.min(axis=0)
        f_max_r, f_max_c = frame_coords.max(axis=0)
        
        H = f_max_r - f_min_r + 1
        W = f_max_c - f_min_c + 1
        output = np.zeros((H, W), dtype=int)
        
        for r, c in frame_coords:
            output[r - f_min_r, c - f_min_c] = input_grid[r, c]
            
        for r, c in seed_coords:
            ro = (r - s_min_r) + 1
            co = (c - s_min_c) + 1
            if 0 <= ro < H and 0 <= co < W:
                output[ro, co] = seed_color
                
        new_output = output.copy()
        rows_map = {}
        cols_map = {}
        for r in range(H):
            for c in range(W):
                color = output[r, c]
                if color != 0 and color != seed_color:
                    if r not in rows_map: rows_map[r] = []
                    rows_map[r].append((c, color))
                    if c not in cols_map: cols_map[c] = []
                    cols_map[c].append((r, color))
                    
        for r in range(H):
            for c in range(W):
                if output[r, c] == seed_color:
                    candidates = [] 
                    if r in rows_map:
                        for fc, fcolor in rows_map[r]:
                            candidates.append((abs(c - fc), fcolor))
                    if c in cols_map:
                        for fr, fcolor in cols_map[c]:
                            candidates.append((abs(r - fr), fcolor))
                            
                    if not candidates:
                        continue
                        
                    min_dist = min(cand[0] for cand in candidates)
                    colors_at_min = set(cand[1] for cand in candidates if cand[0] == min_dist)
                    
                    if len(colors_at_min) == 1:
                        new_output[r, c] = list(colors_at_min)[0]
                    else:
                        new_output[r, c] = seed_color
        return new_output

    test_preds = []
    for inp, out in solver.pairs:
        pred = solve_single(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
    
    for ti in solver.test_in:
        pred = solve_single(ti)
        if pred is None: return None
        test_preds.append(pred)
        
    return test_preds
