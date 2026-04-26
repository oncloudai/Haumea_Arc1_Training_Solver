import numpy as np
from typing import List, Optional

def solve_grid_f1cefba8(input_grid):
    grid = np.array(input_grid)
    rows, cols = grid.shape
    
    unique_colors = np.unique(grid)
    colors = [c for c in unique_colors if c != 0]
    if len(colors) < 2:
        return grid
    
    c1, c2 = colors[0], colors[1]
    
    def get_bbox(c):
        coords = np.argwhere(grid == c)
        if len(coords) == 0: return None
        return coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max()
    
    bbox1 = get_bbox(c1)
    bbox2 = get_bbox(c2)
    
    if bbox1[0] <= bbox2[0] and bbox1[1] >= bbox2[1] and bbox1[2] <= bbox2[2] and bbox1[3] >= bbox2[3]:
        f_color, i_color = c1, c2
    else:
        f_color, i_color = c2, c1
        
    i_coords = np.argwhere(grid == i_color)
    row_counts = {}
    for r, c in i_coords:
        row_counts[r] = row_counts.get(r, 0) + 1
    
    counts = [v for v in row_counts.values() if v > 1]
    if not counts:
        main_count = max(row_counts.values())
    else:
        main_count = max(set(counts), key=counts.count)
        
    main_rows = [r for r, count in row_counts.items() if count >= main_count - 1 and count > 1]
    if not main_rows: return grid
    r_start, r_end = min(main_rows), max(main_rows)
    
    col_spans = []
    for r in main_rows:
        cols_in_row = i_coords[i_coords[:, 0] == r][:, 1]
        col_spans.append((cols_in_row.min(), cols_in_row.max()))
    
    if col_spans:
        c_start, c_end = max(set(col_spans), key=col_spans.count)
    else:
        c_start, c_end = i_coords[:, 1].min(), i_coords[:, 1].max()

    protrusions = []
    for r, c in i_coords:
        if not (r_start <= r <= r_end and c_start <= c <= c_end):
            protrusions.append((r, c))
            
    output_grid = grid.copy()
    f_bbox = get_bbox(f_color)
    frm_r_min, frm_r_max, frm_c_min, frm_c_max = f_bbox

    for pr, pc in protrusions:
        if c_start <= pc <= c_end:
            for r in range(rows):
                if grid[r, pc] == 0:
                    output_grid[r, pc] = i_color
                elif r_start <= r <= r_end and grid[r, pc] == i_color:
                    output_grid[r, pc] = f_color
                elif r == pr and c == pc:
                    output_grid[r, pc] = f_color
                    
        if r_start <= pr <= r_end:
            for c in range(cols):
                if grid[pr, c] == 0:
                    output_grid[pr, c] = i_color
                elif c_start <= c <= c_end and grid[pr, c] == i_color:
                    output_grid[pr, c] = f_color
                elif r == pr and c == pc:
                    output_grid[pr, c] = f_color

    for pr, pc in protrusions:
        if frm_r_min <= pr <= frm_r_max and frm_c_min <= pc <= frm_c_max:
             if output_grid[pr, pc] == i_color:
                 output_grid[pr, pc] = f_color

    return output_grid

def solve_projection_from_protrusions(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_f1cefba8(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_f1cefba8(ti) for ti in solver.test_in]
    return None
