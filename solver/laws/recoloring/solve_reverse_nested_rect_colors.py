import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_reverse_nested_rect_colors(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies all hollow rectangular components in the grid.
    Sorts them by area.
    Reverses the sequence of colors assigned to these rectangles.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        rects = []
        for color in range(1, 10):
            mask = (grid == color).astype(int)
            labeled, num_f = label(mask)
            for i in range(1, num_f + 1):
                comp_mask = (labeled == i)
                rows, cols = np.where(comp_mask)
                r1, r2 = rows.min(), rows.max()
                c1, c2 = cols.min(), cols.max()
                rects.append({
                    'color': color,
                    'r1': r1, 'r2': r2, 'c1': c1, 'c2': c2,
                    'area': (r2 - r1 + 1) * (c2 - c1 + 1)
                })
        
        if not rects: return None
        
        # Sort by area largest to smallest
        rects.sort(key=lambda x: x['area'], reverse=True)
        
        # Original colors in sorted order
        orig_colors = [r['color'] for r in rects]
        # Reversed colors
        new_colors = orig_colors[::-1]
        
        out = np.zeros_like(grid)
        for i, rect in enumerate(rects):
            target_color = new_colors[i]
            r1, r2, c1, c2 = rect['r1'], rect['r2'], rect['c1'], rect['c2']
            # Draw the hollow rectangle frame
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    if r == r1 or r == r2 or c == c1 or c == c2:
                        out[r, c] = target_color
        return out

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
