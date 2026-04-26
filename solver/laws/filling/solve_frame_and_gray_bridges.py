import numpy as np
from typing import List, Optional

def solve_grid_f35d900a(input_grid):
    grid = np.array(input_grid)
    rows, cols = grid.shape
    points = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0: points.append((r, c, grid[r, c]))
    
    if len(points) != 4: return grid
    rows_p = sorted(list(set(p[0] for p in points)))
    cols_p = sorted(list(set(p[1] for p in points)))
    
    if len(rows_p) != 2 or len(cols_p) != 2: return grid
    r1, r2 = rows_p
    c1, c2 = cols_p
    output_grid = grid.copy()
    
    def draw_frame(r, c, color):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if dr == 0 and dc == 0: continue
                    output_grid[nr, nc] = color

    pt_colors = {(p[0], p[1]): p[2] for p in points}
    for r, c, color in points:
        other_c = c1 if c == c2 else c2
        frame_color = pt_colors[(r, other_c)]
        draw_frame(r, c, frame_color)

    for r in [r1, r2]:
        gray_cols = set()
        count = 0
        for c in range(c1 + 2, c2 - 1, 2):
            gray_cols.add(c)
            count += 1
            if count == 2: break
        count = 0
        for c in range(c2 - 2, c1 + 1, -2):
            gray_cols.add(c)
            count += 1
            if count == 2: break
        for c in gray_cols:
            if 0 <= r < rows and 0 <= c < cols: output_grid[r, c] = 5
                    
    for c in [c1, c2]:
        gray_rows = set()
        count = 0
        for r in range(r1 + 2, r2 - 1, 2):
            gray_rows.add(r)
            count += 1
            if count == 2: break
        count = 0
        for r in range(r2 - 2, r1 + 1, -2):
            gray_rows.add(r)
            count += 1
            if count == 2: break
        for r in gray_rows:
            if 0 <= r < rows and 0 <= c < cols: output_grid[r, c] = 5
        
    return output_grid

def solve_frame_and_gray_bridges(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_f35d900a(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_f35d900a(ti) for ti in solver.test_in]
    return None
