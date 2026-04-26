import numpy as np
from typing import List, Optional

def solve_stepped_zigzag_expansion(solver) -> Optional[List[np.ndarray]]:
    for anchor_color in range(1, 10):
        for fill_color in range(1, 10):
            if anchor_color == fill_color: continue
            
            consistent = True; found_any = False
            for inp, out in solver.pairs:
                h, w = inp.shape
                anchors = np.argwhere(inp == anchor_color)
                if len(anchors) != 1: consistent = False; break
                r0, c0 = anchors[0]
                
                pred = inp.copy()
                
                # Upward
                curr_r, curr_c = r0, c0
                while curr_r > 0:
                    if curr_r - 1 >= 0: pred[curr_r - 1, curr_c] = fill_color
                    if curr_r - 2 >= 0:
                        for dc in [0, 1, 2]:
                            if 0 <= curr_c + dc < w: pred[curr_r - 2, curr_c + dc] = fill_color
                    curr_r -= 2; curr_c += 2
                    if curr_c >= w: break
                
                # Downward
                curr_r, curr_c = r0, c0
                while curr_r < h - 1:
                    if curr_r + 1 < h: pred[curr_r + 1, curr_c] = fill_color
                    if curr_r + 2 < h:
                        for dc in [0, -1, -2]:
                            if 0 <= curr_c + dc < w: pred[curr_r + 2, curr_c + dc] = fill_color
                    curr_r += 2; curr_c -= 2
                    if curr_c < 0:
                        # Continue filling downward if r+1 or r+2 are valid
                        if curr_r < h:
                            if curr_r - 1 < h and 0 <= curr_c + 2 < w: # Wait, logic needs to be careful
                                pass
                        break
                
                # Refined downward to handle edge cases like Pair 0 Row 9
                # Let's just use a simple loop
                pred = inp.copy()
                # Up
                cr, cc = r0, c0
                while cr > 0:
                    if cr - 1 >= 0 and 0 <= cc < w: pred[cr-1, cc] = fill_color
                    if cr - 2 >= 0:
                        for dc in [0, 1, 2]:
                            if 0 <= cc + dc < w: pred[cr-2, cc+dc] = fill_color
                    cr -= 2; cc += 2
                # Down
                cr, cc = r0, c0
                while cr < h - 1:
                    if cr + 1 < h and 0 <= cc < w: pred[cr+1, cc] = fill_color
                    if cr + 2 < h:
                        for dc in [0, -1, -2]:
                            if 0 <= cc + dc < w: pred[cr+2, cc+dc] = fill_color
                    cr += 2; cc -= 2
                
                if not np.array_equal(pred, out):
                    consistent = False; break
                found_any = True
                
            if consistent and found_any:
                results = []
                for ti in solver.test_in:
                    h, w = ti.shape; res = ti.copy()
                    anchors = np.argwhere(ti == anchor_color)
                    if len(anchors) == 1:
                        r0, c0 = anchors[0]
                        cr, cc = r0, c0
                        while cr > 0:
                            if cr - 1 >= 0 and 0 <= cc < w: res[cr-1, cc] = fill_color
                            if cr - 2 >= 0:
                                for dc in [0, 1, 2]:
                                    if 0 <= cc + dc < w: res[cr-2, cc+dc] = fill_color
                            cr -= 2; cc += 2
                        cr, cc = r0, c0
                        while cr < h - 1:
                            if cr + 1 < h and 0 <= cc < w: res[cr+1, cc] = fill_color
                            if cr + 2 < h:
                                for dc in [0, -1, -2]:
                                    if 0 <= cc + dc < w: res[cr+2, cc+dc] = fill_color
                            cr += 2; cc -= 2
                    results.append(res)
                return results
    return None
