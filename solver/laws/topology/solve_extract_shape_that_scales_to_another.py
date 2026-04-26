import numpy as np
from typing import List, Optional

def solve_extract_shape_that_scales_to_another(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies all unique non-background colors and their bounding box shapes.
    Checks if any smaller shape, when scaled by a factor (2 or 3), matches or contains 
    another larger shape found in the grid.
    If a match is found, returns the bounding box of the smaller 'seed' shape.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        unique, counts = np.unique(grid, return_counts=True)
        bg = int(unique[np.argmax(counts)])
        colors = [int(c) for c in unique if c != bg]

        # Build binary mask for each color's bounding box
        masks, bboxes = {}, {}
        for c in colors:
            rs, cs = np.where(grid == c)
            if len(rs) == 0: continue
            r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
            bboxes[c] = (r0, r1, c0, c1)
            m = np.zeros((int(r1-r0+1), int(c1-c0+1)), bool)
            m[rs-r0, cs-c0] = True
            masks[c] = m

        for small in colors:
            if small not in masks: continue
            ms = masks[small]
            for k in [2, 3]:
                # Scale the small pattern by k
                scaled = np.kron(ms, np.ones((k, k), bool))
                sh, sw = scaled.shape
                for large in colors:
                    if large == small or large not in masks:
                        continue
                    ml = masks[large]
                    lh, lw = ml.shape
                    # Check every window position in scaled
                    for dr in range(sh):
                        for dc in range(sw):
                            s_r = min(lh, sh - dr)
                            s_c = min(lw, sw - dc)
                            window = np.zeros((lh, lw), bool)
                            window[:s_r, :s_c] = scaled[dr:dr+s_r, dc:dc+s_c]
                            if np.array_equal(window, ml):
                                r0, r1, c0, c1 = bboxes[small]
                                return grid[r0:r1+1, c0:c1+1]

        return None

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
