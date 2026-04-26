
import numpy as np
from typing import List, Optional

def solve_line_between_identical_points(solver) -> Optional[List[np.ndarray]]:
    """
    For each color, if it appears exactly twice in a row or column, draw a solid line between them.
    Vertical lines overwrite horizontal lines.
    """
    for inp, out in solver.pairs:
        res = np.zeros_like(inp)
        colors = np.unique(inp)
        colors = colors[colors != 0]
        lines = []
        for c in colors:
            coords = np.argwhere(inp == c)
            if len(coords) == 2:
                r1, c1 = coords[0]
                r2, c2 = coords[1]
                if r1 == r2:
                    lines.append(('row', r1, min(c1, c2), max(c1, c2), c))
                elif c1 == c2:
                    lines.append(('col', c1, min(r1, r2), max(r1, r2), c))
        
        # Draw rows first
        for type, pos, start, end, color in lines:
            if type == 'row':
                res[pos, start:end+1] = color
        # Draw columns (overwriting)
        for type, pos, start, end, color in lines:
            if type == 'col':
                res[start:end+1, pos] = color
        
        if not np.array_equal(res, out):
            return None
            
    # If consistent across all training pairs
    results = []
    for ti in solver.test_in:
        res = np.zeros_like(ti)
        colors = np.unique(ti)
        colors = colors[colors != 0]
        lines = []
        for c in colors:
            coords = np.argwhere(ti == c)
            if len(coords) == 2:
                r1, c1 = coords[0]
                r2, c2 = coords[1]
                if r1 == r2:
                    lines.append(('row', r1, min(c1, c2), max(c1, c2), c))
                elif c1 == c2:
                    lines.append(('col', c1, min(r1, r2), max(r1, r2), c))
        for type, pos, start, end, color in lines:
            if type == 'row':
                res[pos, start:end+1] = color
        for type, pos, start, end, color in lines:
            if type == 'col':
                res[start:end+1, pos] = color
        results.append(res)
    return results
