import numpy as np
from typing import List, Optional

def solve_marker_3x3_stamping(solver) -> Optional[List[np.ndarray]]:
    for marker_color in range(1, 10):
        for fill_color in range(1, 10):
            consistent = True; found_any = False
            for inp, out in solver.pairs:
                h, w = inp.shape
                pred = np.zeros_like(inp)
                markers = np.argwhere(inp == marker_color)
                if len(markers) == 0: consistent = False; break
                
                for r, c in markers:
                    # Draw 3x3 block centered at (r, c)
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                pred[nr, nc] = fill_color
                                found_any = True
                
                if not np.array_equal(pred, out):
                    consistent = False; break
            
            if consistent and found_any:
                results = []
                for ti in solver.test_in:
                    h, w = ti.shape; res = np.zeros_like(ti)
                    markers = np.argwhere(ti == marker_color)
                    for r, c in markers:
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < h and 0 <= nc < w: res[nr, nc] = fill_color
                    results.append(res)
                return results
    return None
