import numpy as np
from typing import List, Optional

def solve_recolor_bottom_half_of_bars(solver) -> Optional[List[np.ndarray]]:
    for bar_color in range(1, 10):
        for target_color in range(1, 10):
            if bar_color == target_color: continue
            
            consistent = True; found_any = False
            for inp, out in solver.pairs:
                h, w = inp.shape
                pred = inp.copy()
                for c in range(w):
                    col = inp[:, c]
                    bar_pixels = np.argwhere(col == bar_color)
                    if len(bar_pixels) > 0:
                        height = len(bar_pixels)
                        # Check if it's a contiguous bar ending at the same place
                        # (Actually, let's just use the height)
                        num_to_recolor = height // 2
                        if num_to_recolor > 0:
                            # Recolor the bottom-most pixels of the bar
                            bottom_indices = bar_pixels.flatten()[-num_to_recolor:]
                            pred[bottom_indices, c] = target_color
                            found_any = True
                
                if not np.array_equal(pred, out):
                    consistent = False; break
            
            if consistent and found_any:
                results = []
                for ti in solver.test_in:
                    res = ti.copy()
                    for c in range(ti.shape[1]):
                        col = ti[:, c]
                        bar_pixels = np.argwhere(col == bar_color)
                        if len(bar_pixels) > 0:
                            num_to_recolor = len(bar_pixels) // 2
                            if num_to_recolor > 0:
                                bottom_indices = bar_pixels.flatten()[-num_to_recolor:]
                                res[bottom_indices, c] = target_color
                    results.append(res)
                return results
    return None
