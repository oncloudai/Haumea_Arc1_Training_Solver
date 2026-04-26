import numpy as np
from typing import List, Optional
from collections import Counter

def solve_grid_fill_gaps(solver) -> Optional[List[np.ndarray]]:
    for divider_color in [0]:
        consistent = True; found_any = False
        for inp, out in solver.pairs:
            h, w = inp.shape
            div_rows = [r for r in range(h) if np.all(inp[r, :] == divider_color)]
            div_cols = [c for c in range(w) if np.all(inp[:, c] == divider_color)]
            if not div_rows or not div_cols: consistent = False; break
            
            r_bounds = [-1] + div_rows + [h]; c_bounds = [-1] + div_cols + [w]
            NR, NC = len(r_bounds) - 1, len(c_bounds) - 1
            rooms = [[inp[r_bounds[i]+1:r_bounds[i+1], c_bounds[j]+1:c_bounds[j+1]] for j in range(NC)] for i in range(NR)]
            
            # Find default color in rooms
            room_pixels = []
            for r in range(NR):
                for c in range(NC): room_pixels.extend(rooms[r][c].flatten())
            counts = Counter(room_pixels)
            if not counts: consistent = False; break
            default_color = counts.most_common(1)[0][0]
            
            rh, rw = rooms[0][0].shape; pred = inp.copy()
            for lr in range(rh):
                for lc in range(rw):
                    for color in range(1, 10):
                        if color == default_color: continue
                        for r in range(NR):
                            indices = [c for c in range(NC) if rooms[r][c][lr, lc] == color]
                            if len(indices) >= 2:
                                for c in range(min(indices), max(indices) + 1):
                                    gr, gc = r_bounds[r]+1+lr, c_bounds[c]+1+lc
                                    if pred[gr, gc] == default_color:
                                        pred[gr, gc] = color; found_any = True
                        for c in range(NC):
                            indices = [r for r in range(NR) if rooms[r][c][lr, lc] == color]
                            if len(indices) >= 2:
                                for r in range(min(indices), max(indices) + 1):
                                    gr, gc = r_bounds[r]+1+lr, c_bounds[c]+1+lc
                                    if pred[gr, gc] == default_color:
                                        pred[gr, gc] = color; found_any = True
            
            if not np.array_equal(pred, out):
                consistent = False; break
                
        if consistent and found_any:
            results = []
            for ti in solver.test_in:
                h, w = ti.shape
                div_rows = [r for r in range(h) if np.all(ti[r, :] == divider_color)]
                div_cols = [c for c in range(w) if np.all(ti[:, c] == divider_color)]
                r_bounds = [-1] + div_rows + [h]; c_bounds = [-1] + div_cols + [w]
                NR, NC = len(r_bounds)-1, len(c_bounds)-1
                rooms = [[ti[r_bounds[i]+1:r_bounds[i+1], c_bounds[j]+1:c_bounds[j+1]] for j in range(NC)] for i in range(NR)]
                room_pixels = []
                for r in range(NR):
                    for c in range(NC): room_pixels.extend(rooms[r][c].flatten())
                default_color = Counter(room_pixels).most_common(1)[0][0]
                rh, rw = rooms[0][0].shape; res = ti.copy()
                for lr in range(rh):
                    for lc in range(rw):
                        for color in range(1, 10):
                            if color == default_color: continue
                            for r in range(NR):
                                indices = [c for c in range(NC) if rooms[r][c][lr, lc] == color]
                                if len(indices) >= 2:
                                    for c in range(min(indices), max(indices)+1):
                                        gr, gc = r_bounds[r]+1+lr, c_bounds[c]+1+lc
                                        if res[gr, gc] == default_color: res[gr, gc] = color
                            for c in range(NC):
                                indices = [r for r in range(NR) if rooms[r][c][lr, lc] == color]
                                if len(indices) >= 2:
                                    for r in range(min(indices), max(indices)+1):
                                        gr, gc = r_bounds[r]+1+lr, c_bounds[c]+1+lc
                                        if res[gr, gc] == default_color: res[gr, gc] = color
                results.append(res)
            return results
    return None
