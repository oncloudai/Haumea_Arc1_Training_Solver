import numpy as np
from typing import List, Optional

def solve_fill_cells_by_divider_gaps(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a grid of cells separated by a 'dividing' background color.
    Finds gaps (color 0) in the divider lines.
    Fills cells with color 4 if any of their borders have a gap, otherwise color 3.
    Also recolors all gaps in dividers to color 4.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # 1. Identify divider color
        unique, counts = np.unique(grid, return_counts=True)
        colors = {u: c for u, c in zip(unique, counts) if u != 0}
        if not colors: return None
        bg_color = max(colors, key=colors.get)
        
        # 2. Identify divider lines
        row_divs_all = [r for r in range(h) if np.mean(grid[r, :] == bg_color) > 0.5]
        col_divs_all = [c for c in range(w) if np.mean(grid[:, c] == bg_color) > 0.5]
        
        if not row_divs_all or not col_divs_all: return None
        
        def get_groups(indices):
            if not indices: return []
            groups = [[indices[0]]]
            for i in range(1, len(indices)):
                if indices[i] == indices[i-1] + 1: groups[-1].append(indices[i])
                else: groups.append([indices[i]])
            return groups

        row_groups = get_groups(row_divs_all)
        col_groups = get_groups(col_divs_all)
        
        row_starts = [0] + [g[-1] + 1 for g in row_groups]
        row_ends = [g[0] for g in row_groups] + [h]
        row_ints = [(s, e) for s, e in zip(row_starts, row_ends) if e > s]
        
        col_starts = [0] + [g[-1] + 1 for g in col_groups]
        col_ends = [g[0] for g in col_groups] + [w]
        col_ints = [(s, e) for s, e in zip(col_starts, col_ends) if e > s]
        
        num_rows = len(row_ints)
        num_cols = len(col_ints)
        if num_rows < 2 or num_cols < 2: return None
        
        cell_colors = np.full((num_rows, num_cols), 3, dtype=int)
        
        # 3. Fill 0s in dividers with 4 in output
        out = grid.copy()
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 0:
                    out[r, c] = 4
                    
        # 4. Check row dividers for gaps
        for i in range(len(row_groups)):
            div_rows = row_groups[i]
            for j in range(num_cols):
                cs, ce = col_ints[j]
                if np.any(grid[div_rows[0]:div_rows[-1]+1, cs:ce] == 0):
                    # Gap in divider between row-cell i and i+1
                    cell_colors[i, j] = 4
                    cell_colors[i+1, j] = 4
                    
        # 5. Check col dividers for gaps
        for j in range(len(col_groups)):
            div_cols = col_groups[j]
            for i in range(num_rows):
                rs, re = row_ints[i]
                if np.any(grid[rs:re, div_cols[0]:div_cols[-1]+1] == 0):
                    # Gap in divider between col-cell j and j+1
                    cell_colors[i, j] = 4
                    cell_colors[i, j+1] = 4
                    
        # 6. Apply cell colors
        for i in range(num_rows):
            for j in range(num_cols):
                rs, re = row_ints[i]
                cs, ce = col_ints[j]
                out[rs:re, cs:ce] = cell_colors[i, j]
                
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
