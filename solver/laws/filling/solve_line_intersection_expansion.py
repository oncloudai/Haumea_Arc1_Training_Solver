import numpy as np
from typing import List, Optional

def solve_line_intersection_expansion(solver) -> Optional[List[np.ndarray]]:
    # Task bdad9b1f
    bg = 0
    # Infer rules: which color expands in which direction, and what is the intersection color?
    # Colors involved in expansion
    color_rules = {} # color -> 'H' or 'V'
    intersection_map = {} # (color1, color2) -> intersection_color
    
    for inp, out in solver.pairs:
        h, w = inp.shape
        unq_in = np.unique(inp[inp != bg])
        unq_out = np.unique(out[out != bg])
        
        new_colors = [c for c in unq_out if c not in unq_in]
        if not new_colors: continue
        int_color = new_colors[0]
        
        for c in unq_in:
            in_coords = np.argwhere(inp == c)
            out_coords = np.argwhere(out == c)
            # Check if it became a full row or column
            is_V = False; is_H = False
            for r, col in in_coords:
                if np.all((out[:, col] == c) | (out[:, col] == int_color)): is_V = True
                if np.all((out[r, :] == c) | (out[r, :] == int_color)): is_H = True
            
            if is_V: color_rules[c] = 'V'
            if is_H: color_rules[c] = 'H'
            
        # Determine intersection map
        h_colors = [c for c, d in color_rules.items() if d == 'H']
        v_colors = [c for c, d in color_rules.items() if d == 'V']
        if h_colors and v_colors:
            intersection_map[(h_colors[0], v_colors[0])] = int_color

    if not color_rules or not intersection_map: return None

    def process(grid):
        h, w = grid.shape; res = np.full_like(grid, bg)
        h_colors = [c for c, d in color_rules.items() if d == 'H']
        v_colors = [c for c, d in color_rules.items() if d == 'V']
        
        # Draw lines
        h_rows = []
        for hc in h_colors:
            rows = np.unique(np.argwhere(grid == hc)[:, 0])
            for r in rows:
                res[r, :] = hc
                h_rows.append(r)
                
        v_cols = []
        for vc in v_colors:
            cols = np.unique(np.argwhere(grid == vc)[:, 1])
            for c in cols:
                res[:, c] = vc
                v_cols.append(c)
                
        # Draw intersections
        for hc, vc in intersection_map.keys():
            int_c = intersection_map[(hc, vc)]
            rows = np.unique(np.argwhere(grid == hc)[:, 0])
            cols = np.unique(np.argwhere(grid == vc)[:, 1])
            for r in rows:
                for c in cols:
                    res[r, c] = int_c
        return res

    for inp, out in solver.pairs:
        if not np.array_equal(process(inp), out): return None
        
    return [process(ti) for ti in solver.test_in]
