import numpy as np
from typing import List, Optional

def solve_bounce_diagonal_from_middle(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        consistent = True
        found_any = False
        for inp, out in solver.pairs:
            h, w = inp.shape
            if inp.shape != out.shape: consistent = False; break
            
            # Find candidate color in the middle of the last row
            # Actually, it's the color that is different from its neighbors in the last row
            last_row = inp[h-1, :]
            unq, counts = np.unique(last_row, return_counts=True)
            # Dominant color in last row is the "base"
            if len(unq) < 2: consistent = False; break
            base_color = unq[np.argmax(counts)]
            others = [c for c in unq if c != base_color and c != bg]
            if not others: consistent = False; break
            middle_color = others[0]
            
            # Find the position(s) of middle_color in the last row
            pos = np.where(last_row == middle_color)[0]
            if len(pos) == 0: consistent = False; break
            
            pred = inp.copy()
            for p in pos:
                # Up and left
                r, c = h-1, p
                while r > 0 and c > 0:
                    r -= 1; c -= 1
                    pred[r, c] = middle_color
                # Up and right
                r, c = h-1, p
                while r > 0 and c < w - 1:
                    r -= 1; c += 1
                    pred[r, c] = middle_color
            
            # Wait, in b8cdaf2b, it's not the whole path.
            # It's only the pixels that are NOT already filled by something else?
            # Or maybe it stops at some point.
            # Let's re-examine Example 1:
            # Input: (4,2) is 3. Last row is 8 8 3 8 8.
            # Output: (1,0)=3, (1,4)=3, (2,1)=3, (2,3)=3.
            # (4,2) -> (3,1), (3,3) are occupied by 8?
            # IN 1: (3,2)=8. (4,0)=8, (4,1)=8, (4,2)=3, (4,3)=8, (4,4)=8.
            # Output 1: (2,1)=3, (2,3)=3. (1,0)=3, (1,4)=3.
            # So from (4,2), it skips row 3 (occupied by 8) and fills 3 in rows 2 and 1.
            
            # Correct logic: for each row r < h-1, if the diagonal from (h-1, p) is NOT occupied by any other color, fill it.
            # Actually, it seems it fills it regardless but only if it's currently bg (0).
            pred = inp.copy()
            for p in pos:
                for dr, dc in [(-1, -1), (-1, 1)]:
                    curr_r, curr_c = h-1, p
                    while True:
                        curr_r += dr; curr_c += dc
                        if 0 <= curr_r < h and 0 <= curr_c < w:
                            if pred[curr_r, curr_c] == bg:
                                pred[curr_r, curr_c] = middle_color
                            else:
                                # Stop if we hit a non-bg pixel?
                                break
                        else: break
            
            if not np.array_equal(pred, out):
                consistent = False; break
            found_any = True
            
        if consistent and found_any:
            def process(grid):
                h, w = grid.shape
                last_row = grid[h-1, :]
                unq, counts = np.unique(last_row, return_counts=True)
                if len(unq) < 2: return grid
                base_color = unq[np.argmax(counts)]
                others = [c for c in unq if c != base_color and c != bg]
                if not others: return grid
                middle_color = others[0]
                pos = np.where(last_row == middle_color)[0]
                res = grid.copy()
                for p in pos:
                    for dr, dc in [(-1, -1), (-1, 1)]:
                        curr_r, curr_c = h-1, p
                        while True:
                            curr_r += dr; curr_c += dc
                            if 0 <= curr_r < h and 0 <= curr_c < w:
                                if res[curr_r, curr_c] == bg: res[curr_r, curr_c] = middle_color
                                else: break
                            else: break
                return res
            return [process(ti) for ti in solver.test_in]
    return None
