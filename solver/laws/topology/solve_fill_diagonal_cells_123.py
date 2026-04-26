import numpy as np
from typing import List, Optional

def solve_fill_diagonal_cells_123(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a grid structure defined by color 5 lines.
    Finds the grid of cells (regions of 0s).
    Fills three specific cells along the 'diagonal' with colors 1, 2, and 3.
    The cells are: (0, 0), (N//2, M//2), and (N-1, M-1).
    """
    def get_intervals(h, w, grid, axis):
        # A divider is a row/col that is mostly color 5
        divs = []
        for i in range(h if axis==0 else w):
            line = grid[i, :] if axis==0 else grid[:, i]
            # Check if line is mostly 5s (allowing for some noise or gaps)
            if np.sum(line == 5) >= (len(line) // 2 + 1):
                divs.append(i)
        
        if not divs: return []
        
        # Group contiguous boundaries
        groups = []
        if divs:
            groups.append([divs[0]])
            for i in range(1, len(divs)):
                if divs[i] == divs[i-1] + 1:
                    groups[-1].append(divs[i])
                else:
                    groups.append([divs[i]])
        
        # Starts and ends of cell regions
        starts = [0] + [g[-1] + 1 for g in groups]
        ends = [g[0] for g in groups] + [h if axis==0 else w]
        
        intervals = []
        for s, e in zip(starts, ends):
            if e > s:
                intervals.append((s, e))
        return intervals

    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        row_intervals = get_intervals(h, w, grid, 0)
        col_intervals = get_intervals(h, w, grid, 1)
        
        if not row_intervals or not col_intervals: return None
        
        num_rows = len(row_intervals)
        num_cols = len(col_intervals)
        
        out = grid.copy()
        
        # Mapping rules from observations:
        # First cell -> color 1
        # Middle cell -> color 2
        # Last cell -> color 3
        # Middle is num_rows // 2
        
        indices = [
            (0, 0, 1),
            (num_rows // 2, num_cols // 2, 2),
            (num_rows - 1, num_cols - 1, 3)
        ]
        
        # Sort out duplicates
        final_indices = []
        seen_cells = set()
        for ri, ci, color in indices:
            if (ri, ci) not in seen_cells:
                final_indices.append((ri, ci, color))
                seen_cells.add((ri, ci))
        
        for ri, ci, color in final_indices:
            r_start, r_end = row_intervals[ri]
            c_start, c_end = col_intervals[ci]
            out[r_start:r_end, c_start:c_end] = color
            
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
