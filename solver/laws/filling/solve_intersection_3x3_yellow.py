
import numpy as np
from typing import List, Optional

def solve_intersection_3x3_yellow(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies two orthogonal lines (one full row and one full column, or similar).
    Finds their intersection point (r, c).
    Draws a 3x3 square of color 4 centered at (r, c).
    The pixel at (r, c) itself retains its original color.
    """
    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        
        # Find horizontal lines (color != 0, length > 2)
        h_lines = []
        for r in range(rows):
            unique = np.unique(grid[r, :])
            colors = unique[unique != 0]
            for color in colors:
                # Check if it's a line
                coords = np.argwhere(grid[r, :] == color)
                if len(coords) >= 3:
                    h_lines.append({'row': r, 'color': color})
                    
        # Find vertical lines
        v_lines = []
        for c in range(cols):
            unique = np.unique(grid[:, c])
            colors = unique[unique != 0]
            for color in colors:
                coords = np.argwhere(grid[:, c] == color)
                if len(coords) >= 3:
                    v_lines.append({'col': c, 'color': color})
                    
        if not h_lines or not v_lines: return None
        
        output = grid.copy()
        found_any = False
        for hl in h_lines:
            for vl in v_lines:
                # Intersection point
                r, c = hl['row'], vl['col']
                # Draw 3x3 around (r, c)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if dr == 0 and dc == 0:
                                # Retain original color
                                pass
                            else:
                                if output[nr, nc] != 4:
                                    output[nr, nc] = 4
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
