
import numpy as np
from typing import List, Optional

def solve_extend_red_crosses_with_azure(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies Red (2) horizontal and vertical segments of length > 1.
    For each intersecting pair of segments, draws an Azure (8) cross.
    Radius D is the max distance from the intersection to the segment boundaries.
    """
    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        red_coords = np.argwhere(grid == 2)
        if len(red_coords) == 0: return None
        
        # 1. Find all horizontal segments (length > 1)
        h_segs = []
        for r in range(rows):
            cols_r = sorted([c for rr, c in red_coords if rr == r])
            if len(cols_r) > 1:
                # Group into contiguous parts? 
                # In this task, segments seem to be simple spans.
                # Let's check for contiguous groups just in case.
                start = 0
                for i in range(1, len(cols_r)):
                    if cols_r[i] > cols_r[i-1] + 1:
                        if i - start > 1:
                            h_segs.append({'row': r, 'min': cols_r[start], 'max': cols_r[i-1]})
                        start = i
                if len(cols_r) - start > 1:
                    h_segs.append({'row': r, 'min': cols_r[start], 'max': cols_r[-1]})
                    
        # 2. Find all vertical segments (length > 1)
        v_segs = []
        for c in range(cols):
            rows_c = sorted([r for rr, cc in red_coords if cc == c])
            if len(rows_c) > 1:
                start = 0
                for i in range(1, len(rows_c)):
                    if rows_c[i] > rows_c[i-1] + 1:
                        if i - start > 1:
                            v_segs.append({'col': c, 'min': rows_c[start], 'max': rows_c[i-1]})
                        start = i
                if len(rows_c) - start > 1:
                    v_segs.append({'col': c, 'min': rows_c[start], 'max': rows_c[-1]})
                    
        if not h_segs or not v_segs: return None
        
        output = grid.copy()
        found_any = False
        
        # 3. Intersections
        for hs in h_segs:
            for vs in v_segs:
                r, c = hs['row'], vs['col']
                # Check intersection condition
                if (hs['min'] <= c <= hs['max']) and (vs['min'] <= r <= vs['max']):
                    # Intersection (r, c)
                    D = max(r - vs['min'], vs['max'] - r, c - hs['min'], hs['max'] - c)
                    
                    # Draw cross
                    for j in range(c - D, c + D + 1):
                        if 0 <= r < rows and 0 <= j < cols:
                            if output[r, j] == 0 or output[r, j] == 5:
                                output[r, j] = 8; found_any = True
                    for i in range(r - D, r + D + 1):
                        if 0 <= i < rows and 0 <= c < cols:
                            if output[i, c] == 0 or output[i, c] == 5:
                                output[i, c] = 8; found_any = True
                                
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
