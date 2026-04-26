import numpy as np
from typing import List, Optional

def solve_dual_red_line_azure_projection(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies two vertical red lines (color 2).
    For each azure (8) marker:
    1. Finds which red line covers the marker's row. That's the 'source' line.
    2. Fills the gap between the source line and the marker with color 8.
    3. Changes the marker color to 4.
    4. Projects to the OTHER red line: finds the corresponding row and fills it
       (excluding the red pixel) with color 8.
    """
    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        
        # 1. Find the two red vertical lines
        red_cols = []
        for c in range(cols):
            if np.any(grid[:, c] == 2):
                coords = np.argwhere(grid[:, c] == 2)
                if len(coords) >= 2:
                    red_cols.append({'col': c, 'rows': (coords.min(), coords.max())})
        
        if len(red_cols) != 2: return None
        
        azure_coords = np.argwhere(grid == 8)
        if len(azure_coords) == 0: return None
        
        output = grid.copy()
        
        for ar, ac in azure_coords:
            # Find which red line has this row in its range
            src_idx = -1
            for i, rc in enumerate(red_cols):
                if rc['rows'][0] <= ar <= rc['rows'][1]:
                    src_idx = i; break
            
            if src_idx == -1: continue
            
            src_rc = red_cols[src_idx]
            dst_rc = red_cols[1 - src_idx]
            
            # 1. Fill gap
            step = 1 if ac > src_rc['col'] else -1
            for c in range(src_rc['col'] + step, ac, step):
                output[ar, c] = 8
            
            # 2. Change marker to 4
            output[ar, ac] = 4
            
            # 3. Project
            row_offset = ar - src_rc['rows'][0]
            target_row = dst_rc['rows'][0] + row_offset
            
            if 0 <= target_row < rows:
                for c in range(cols):
                    if c != dst_rc['col']:
                        output[target_row, c] = 8
                        
        return output

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
