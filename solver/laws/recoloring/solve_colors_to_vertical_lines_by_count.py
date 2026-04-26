import numpy as np
from typing import List, Optional

def solve_colors_to_vertical_lines_by_count(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies all unique non-zero colors in the input and counts their pixels.
    The output width is the number of unique colors.
    The output height is the maximum pixel count.
    Colors are sorted by their pixel counts in descending order.
    Each column in the output is a vertical line of a color with length equal to its count.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        unique_colors = np.unique(grid)
        colors = [c for c in unique_colors if c != 0]
        if not colors: return None
        
        counts = {c: np.sum(grid == c) for c in colors}
        # Sort colors by count (descending). Tie-break by color value if needed.
        sorted_colors = sorted(colors, key=lambda x: (counts[x], x), reverse=True)
        
        max_count = counts[sorted_colors[0]]
        out_w = len(sorted_colors)
        out_h = max_count
        
        out = np.zeros((out_h, out_w), dtype=int)
        for i, color in enumerate(sorted_colors):
            count = counts[color]
            out[0:count, i] = color
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
