import numpy as np
from typing import List, Optional
from collections import Counter

def solve_recolor_strips_by_majority(solver) -> Optional[List[np.ndarray]]:
    """
    Calculates majority colors for each row and each column.
    Decides between horizontal or vertical strips based on which has more agreement (score).
    Recolors the entire grid with those strips.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # Calculate scores for horizontal strips
        row_majorities = []
        h_score = 0
        for r in range(h):
            counts = Counter(grid[r])
            if 0 in counts: del counts[0]
            if not counts:
                row_majorities.append(0)
            else:
                color, count = counts.most_common(1)[0]
                row_majorities.append(color)
                h_score += count
                
        # Calculate scores for vertical strips
        col_majorities = []
        v_score = 0
        for c in range(w):
            counts = Counter(grid[:, c])
            if 0 in counts: del counts[0]
            if not counts:
                col_majorities.append(0)
            else:
                color, count = counts.most_common(1)[0]
                col_majorities.append(color)
                v_score += count
                
        output_grid = np.zeros_like(grid)
        if v_score >= h_score:
            # Vertical strips
            for c in range(w):
                output_grid[:, c] = col_majorities[c]
        else:
            # Horizontal strips
            for r in range(h):
                output_grid[r, :] = row_majorities[r]
        return output_grid

    for inp, out in solver.pairs:
        if not np.array_equal(apply_logic(inp), out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
