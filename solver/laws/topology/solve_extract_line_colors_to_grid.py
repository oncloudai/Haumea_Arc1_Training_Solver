import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_extract_line_colors_to_grid(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a square block of color 5. Its size S determines output SxS.
    Identifies all other non-zero colors that form full-width or full-height lines.
    The colors of these lines form the rows (if horizontal) or columns (if vertical)
    of the output SxS grid.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # 1. Find color 5 block size
        mask_5 = (grid == 5).astype(int)
        labeled, num_f = label(mask_5)
        S = -1
        for i in range(1, num_f + 1):
            comp = (labeled == i)
            rows, cols = np.where(comp)
            r1, r2 = rows.min(), rows.max()
            c1, c2 = cols.min(), cols.max()
            sh, sw = r2 - r1 + 1, c2 - c1 + 1
            if sh == sw and sh > 1:
                S = sh
                break
        
        if S == -1: return None
        
        # 2. Find lines
        # A line is a row or col that has mostly one color
        h_lines = [] # (row_index, color)
        v_lines = [] # (col_index, color)
        
        # Check all non-zero colors except 5
        other_colors = [c for c in np.unique(grid) if c != 0 and c != 5]
        
        for color in other_colors:
            rows, cols = np.where(grid == color)
            # Check if horizontal line
            for r in np.unique(rows):
                if np.sum(grid[r, :] == color) >= min(w, S): # at least S pixels
                    h_lines.append((r, color))
            # Check if vertical line
            for c in np.unique(cols):
                if np.sum(grid[:, c] == color) >= min(h, S):
                    v_lines.append((c, color))
                    
        # Dedup and sort
        h_lines = sorted(list(set(h_lines)))
        v_lines = sorted(list(set(v_lines)))
        
        out = np.zeros((S, S), dtype=int)
        if len(h_lines) == S:
            for i in range(S):
                out[i, :] = h_lines[i][1]
        elif len(v_lines) == S:
            for j in range(S):
                out[:, j] = v_lines[j][1]
        else:
            return None
            
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
