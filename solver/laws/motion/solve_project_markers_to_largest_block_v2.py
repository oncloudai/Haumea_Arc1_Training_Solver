
import numpy as np
from typing import List, Optional

def solve_project_markers_to_largest_block_v2(solver) -> Optional[List[np.ndarray]]:
    """
    Project markers to the boundaries of the largest rectangular block.
    """
    for P in range(1, 10):
        consistent = True
        for inp, out in solver.pairs:
            h, w = inp.shape
            
            # Find the largest rectangular block of a uniform color
            best_block = None; best_area = 0; block_color = -1
            for color in range(1, 10):
                mask = (inp == color)
                if not np.any(mask): continue
                coords = np.argwhere(mask)
                r_min, c_min = coords.min(axis=0); r_max, c_max = coords.max(axis=0)
                area = (r_max - r_min + 1) * (c_max - c_min + 1)
                if area > best_area and np.all(inp[r_min:r_max+1, c_min:c_max+1] == color):
                    best_area = area; best_block = (r_min, c_min, r_max, c_max); block_color = color
            if not best_block: consistent = False; break
            
            res = inp.copy()
            r_min, c_min, r_max, c_max = best_block
            markers = np.argwhere((inp != 0) & (inp != block_color))
            
            # Clear markers
            for r, c in markers: res[r, c] = 0
            
            # Project markers
            for r, c in markers:
                nr, nc = r, c
                if r < r_min:
                    dist = r_min - r
                    new_dist = (dist - 1) % P + 1
                    nr = r_min - new_dist
                elif r > r_max:
                    dist = r - r_max
                    new_dist = (dist - 1) % P + 1
                    nr = r_max + new_dist
                elif c < c_min:
                    dist = c_min - c
                    new_dist = (dist - 1) % P + 1
                    nc = c_min - new_dist
                elif c > c_max:
                    dist = c - c_max
                    new_dist = (dist - 1) % P + 1
                    nc = c_max + new_dist
                
                if 0 <= nr < h and 0 <= nc < w:
                    res[nr, nc] = block_color
            
            if not np.array_equal(res, out):
                consistent = False; break
                
        if consistent:
            results = []
            for ti in solver.test_in:
                h, w = ti.shape
                best_block = None; best_area = 0; block_color = -1
                for color in range(1, 10):
                    mask = (ti == color)
                    if not np.any(mask): continue
                    coords = np.argwhere(mask); r_min, c_min = coords.min(axis=0); r_max, c_max = coords.max(axis=0)
                    area = (r_max - r_min + 1) * (c_max - c_min + 1)
                    if area > best_area and np.all(ti[r_min:r_max+1, c_min:c_max+1] == color):
                        best_area = area; best_block = (r_min, c_min, r_max, c_max); block_color = color
                if not best_block: results.append(ti); continue
                res = ti.copy(); r_min, c_min, r_max, c_max = best_block
                markers = np.argwhere((ti != 0) & (ti != block_color))
                for r, c in markers: res[r, c] = 0
                for r, c in markers:
                    nr, nc = r, c
                    if r < r_min: nr = r_min - ((r_min - r - 1) % P + 1)
                    elif r > r_max: nr = r_max + ((r - r_max - 1) % P + 1)
                    elif c < c_min: nc = c_min - ((c_min - c - 1) % P + 1)
                    elif c > c_max: nc = c_max + ((c - c_max - 1) % P + 1)
                    if 0 <= nr < h and 0 <= nc < w: res[nr, nc] = block_color
                results.append(res)
            return results
    return None
