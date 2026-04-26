import numpy as np
from typing import List, Optional

def solve_move_to_bbox_proximity(solver) -> Optional[List[np.ndarray]]:
    # Try all pairs of colors
    for target_color in range(1, 10):
        for seed_color in range(1, 10):
            if target_color == seed_color: continue
            
            consistent = True; found_any = False
            for inp, out in solver.pairs:
                target_coords = np.argwhere(inp == target_color)
                seed_coords = np.argwhere(inp == seed_color)
                
                if len(target_coords) == 0 or len(seed_coords) == 0:
                    consistent = False; break
                
                # Assume the target is a single object (like the 2x2 square)
                r_min, c_min = target_coords.min(axis=0)
                r_max, c_max = target_coords.max(axis=0)
                
                pred = inp.copy()
                # Remove old seeds
                for r, c in seed_coords:
                    if pred[r, c] == seed_color: pred[r, c] = 0
                
                # Move each seed
                for r, c in seed_coords:
                    nr, nc = r, c
                    if r < r_min: nr = r_min - 1
                    elif r > r_max: nr = r_max + 1
                    
                    if c < c_min: nc = c_min - 1
                    elif c > c_max: nc = c_max + 1
                    
                    if 0 <= nr < pred.shape[0] and 0 <= nc < pred.shape[1]:
                        pred[nr, nc] = seed_color
                
                if not np.array_equal(pred, out):
                    consistent = False; break
                found_any = True
                
            if consistent and found_any:
                results = []
                for ti in solver.test_in:
                    target_coords = np.argwhere(ti == target_color)
                    seed_coords = np.argwhere(ti == seed_color)
                    if len(target_coords) == 0 or len(seed_coords) == 0:
                        results.append(ti.copy()); continue
                    
                    r_min, c_min = target_coords.min(axis=0)
                    r_max, c_max = target_coords.max(axis=0)
                    res = ti.copy()
                    for r, c in seed_coords:
                        if res[r, c] == seed_color: res[r, c] = 0
                    for r, c in seed_coords:
                        nr, nc = r, c
                        if r < r_min: nr = r_min - 1
                        elif r > r_max: nr = r_max + 1
                        if c < c_min: nc = c_min - 1
                        elif c > c_max: nc = c_max + 1
                        if 0 <= nr < res.shape[0] and 0 <= nc < res.shape[1]:
                            res[nr, nc] = seed_color
                    results.append(res)
                return results
    return None
