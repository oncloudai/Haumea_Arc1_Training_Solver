
import numpy as np
from typing import List, Optional
from collections import defaultdict

def solve_push_to_block_with_stacking(solver) -> Optional[List[np.ndarray]]:
    """
    Finds the most frequent non-zero color (usually 5) as the attractor block.
    Markers (other colors) push towards this block along the dimension where
    they are within the block's range. If multiple markers push towards the same
    spot, they stack (pile up).
    """
    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        
        # 1. Identify the attractor color (prefer color 5, otherwise most frequent)
        unique, counts = np.unique(grid[grid != 0], return_counts=True)
        if len(unique) == 0: return None
        attractor_color = 5 if 5 in unique else unique[np.argmax(counts)]
            
        attractor_coords = np.argwhere(grid == attractor_color)
        if len(attractor_coords) == 0: return None
        
        r_min, c_min = attractor_coords.min(axis=0)
        r_max, c_max = attractor_coords.max(axis=0)
        
        # 2. Identify markers (non-zero, non-attractor)
        marker_coords = np.argwhere((grid != 0) & (grid != attractor_color))
        if len(marker_coords) == 0: return None
        
        output = grid.copy()
        # Clear markers from output
        for r, c in marker_coords:
            output[r, c] = 0
            
        # 3. Group markers by push direction and perpendicular coordinate
        above = defaultdict(list)
        below = defaultdict(list)
        left = defaultdict(list)
        right = defaultdict(list)
        
        for r, c in marker_coords:
            if c_min <= c <= c_max:
                if r < r_min: above[c].append(r)
                elif r > r_max: below[c].append(r)
            elif r_min <= r <= r_max:
                if c < c_min: left[r].append(c)
                elif c > c_max: right[r].append(c)
                
        # 4. Place markers with stacking
        changed = False
        for c, rs in above.items():
            rs.sort(reverse=True)
            for i, r in enumerate(rs):
                nr = r_min - 1 - i
                if 0 <= nr < rows:
                    if output[nr, c] != attractor_color:
                        output[nr, c] = attractor_color
                        changed = True
        for c, rs in below.items():
            rs.sort()
            for i, r in enumerate(rs):
                nr = r_max + 1 + i
                if 0 <= nr < rows:
                    if output[nr, c] != attractor_color:
                        output[nr, c] = attractor_color
                        changed = True
        for r, cs in left.items():
            cs.sort(reverse=True)
            for i, c in enumerate(cs):
                nc = c_min - 1 - i
                if 0 <= nc < cols:
                    if output[r, nc] != attractor_color:
                        output[r, nc] = attractor_color
                        changed = True
        for r, cs in right.items():
            cs.sort()
            for i, c in enumerate(cs):
                nc = c_max + 1 + i
                if 0 <= nc < cols:
                    if output[r, nc] != attractor_color:
                        output[r, nc] = attractor_color
                        changed = True
                        
        return output if changed else None

    # Verify on training pairs
    for inp, out in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out):
            return None
            
    # Generate test predictions
    test_preds = []
    for inp in solver.test_in:
        pred = run_single(inp)
        test_preds.append(pred if pred is not None else inp.copy())
            
    return test_preds
