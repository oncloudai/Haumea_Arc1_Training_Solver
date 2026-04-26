
import numpy as np
from typing import List, Optional

def solve_project_rays_from_object_to_anchors(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a 'seed' object (color 9) and multiple 'anchors' (color 8).
    For each pixel in the seed object, draws a ray towards any anchor
    that is in the same row or column.
    The ray uses the anchor's color.
    """
    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        
        # 1. Identify seed pixels
        seed_coords = np.argwhere(grid == 9)
        if len(seed_coords) == 0: return None
        
        # 2. Identify anchor pixels
        anchor_coords = np.argwhere(grid == 8)
        if len(anchor_coords) == 0: return None
        
        output = grid.copy()
        found_any = False
        
        for sr, sc in seed_coords:
            for ar, ac in anchor_coords:
                if sr == ar:
                    # Same row - horizontal ray
                    step = 1 if ac > sc else -1
                    for c in range(sc + step, ac, step):
                        output[sr, c] = 8
                        found_any = True
                if sc == ac:
                    # Same column - vertical ray
                    step = 1 if ar > sr else -1
                    for r in range(sr + step, ar, step):
                        output[r, sc] = 8
                        found_any = True
                        
        return output if found_any else None

    # Verify on pairs
    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    test_preds = []
    for inp in solver.test_in:
        pred = run_single(inp)
        test_preds.append(pred if pred is not None else np.array(inp))
        
    return test_preds
