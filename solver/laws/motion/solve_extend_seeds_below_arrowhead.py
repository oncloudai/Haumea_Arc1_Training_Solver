
import numpy as np
from typing import List, Optional

def solve_extend_seeds_below_arrowhead(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies the 'arrowhead' object (the largest non-zero component).
    Identifies the 'marker' color (the other most frequent non-zero color).
    Finds the tip of the arrowhead.
    Feature columns are tip_col - 2 to tip_col + 2.
    For each feature column, if a marker exists below the arrowhead's base (tip_row + 2),
    fills that column from the bottom of the arrowhead to the grid boundary.
    """
    def get_objects(grid, bg):
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        objs = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != bg and not visited[r, c]:
                    q = [(r, c)]; visited[r, c] = True
                    coords = []
                    while q:
                        cr, cc = q.pop(0); coords.append((cr, cc))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1), (1,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<rows and 0<=nc<cols and grid[nr,nc]!=bg and not visited[nr,nc]:
                                visited[nr,nc]=True; q.append((nr,nc))
                    objs.append(np.array(coords))
        return objs

    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        unique, counts = np.unique(grid, return_counts=True)
        if len(unique) < 2: return None
        bg = unique[np.argmax(counts)]
        
        objs = get_objects(grid, bg)
        if not objs: return None
        
        # Arrowhead is the largest object
        objs.sort(key=len, reverse=True)
        arrow_coords = objs[0]
        arrow_color = grid[arrow_coords[0,0], arrow_coords[0,1]]
        
        # Tip is at minimum row
        tip_row = arrow_coords[:, 0].min()
        # Columns at tip row
        tip_cols = arrow_coords[arrow_coords[:, 0] == tip_row, 1]
        tip_col = int(np.mean(tip_cols))
        
        # Marker color is the other color(s)
        marker_colors = [c for c in unique if c != bg and c != arrow_color]
        if not marker_colors: return None
        m_color = marker_colors[0] # Assume one marker color for the fill
        
        # Feature columns
        feat_cols = range(tip_col - 2, tip_col + 3)
        h_row = tip_row + 2
        
        output = grid.copy()
        found_any = False
        
        for c in feat_cols:
            if c < 0 or c >= cols: continue
            
            # Check for marker in this column below H
            has_marker = False
            for r in range(h_row + 1, rows):
                if grid[r, c] in marker_colors:
                    has_marker = True; break
            
            if has_marker:
                # Find r_start: lowest arrow pixel in this column
                arrow_in_col = arrow_coords[arrow_coords[:, 1] == c, 0]
                if len(arrow_in_col) > 0:
                    r_start = arrow_in_col.max()
                else:
                    # Column is the hole or part of the spread
                    # Use hole row
                    r_start = h_row - 1
                
                # Fill downwards
                for r in range(r_start + 1, rows):
                    if output[r, c] != arrow_color:
                        if output[r, c] != m_color:
                            output[r, c] = m_color
                            found_any = True
                            
        return output if found_any else None

    # Verify
    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    return [run_single(ti) for ti in solver.test_in]
