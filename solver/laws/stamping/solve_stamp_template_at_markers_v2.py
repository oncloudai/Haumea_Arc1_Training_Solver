
import numpy as np
from typing import List, Optional

def solve_stamp_template_at_markers_v2(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a 'template' object (containing color 8 and marker colors 2, 3).
    Finds all other marker pixels in the grid.
    Stamps the template at each marker such that the corresponding marker color aligns.
    """
    def get_objects(grid):
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        objs = []
        # Find the unique object containing 8
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 1 and not visited[r, c]: # 1 is bg
                    q = [(r, c)]; visited[r, c] = True
                    coords = []
                    while q:
                        cr, cc = q.pop(0)
                        coords.append((cr, cc))
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<rows and 0<=nc<cols and grid[nr,nc]!=1 and not visited[nr,nc]:
                                visited[nr,nc]=True; q.append((nr,nc))
                    objs.append(np.array(coords))
        return objs

    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        
        objs = get_objects(grid)
        if not objs: return None
        
        # Template is the object containing color 8
        template_obj = None
        for obj in objs:
            for r, c in obj:
                if grid[r, c] == 8:
                    template_obj = obj; break
            if template_obj is not None: break
            
        if template_obj is None: return None
        
        # Extract template as (dr, dc, color) list relative to its markers
        r_min, c_min = template_obj.min(axis=0)
        r_max, c_max = template_obj.max(axis=0)
        
        template_pixels = []
        markers_in_template = {} # color -> list of (dr, dc)
        
        for r, c in template_obj:
            color = grid[r, c]
            dr, dc = r - r_min, c - c_min
            template_pixels.append((dr, dc, color))
            if color in [2, 3]:
                if color not in markers_in_template: markers_in_template[color] = []
                markers_in_template[color].append((dr, dc))
                
        output = grid.copy()
        
        # Find all markers in the whole grid
        for r in range(rows):
            for c in range(cols):
                color = grid[r, c]
                if color in [2, 3]:
                    # Check if this marker is part of the original template
                    is_in_template = False
                    for tr, tc in template_obj:
                        if r == tr and c == tc:
                            is_in_template = True; break
                    if is_in_template: continue
                    
                    # Stamp template aligned at this marker
                    # If the template has multiple pixels of this color, which one to use?
                    # In Ex 0, some markers are 2x2.
                    # This suggests we should use the TOP-LEFT of the marker group?
                    # Or just try each marker pixel independently.
                    
                    # For a single marker pixel at (r, c):
                    # It matches a marker of the same color in the template at (dr_m, dc_m).
                    # So the template's (0, 0) should be at (r - dr_m, c - dc_m).
                    
                    # To handle 2x2 markers, let's find the object this marker belongs to.
                    # Then use its top-left.
                    pass
        
        # Re-implementation of stamping:
        # Find all non-template objects of colors 2, 3
        marker_objs = []
        for obj in objs:
            is_template = False
            for r, c in obj:
                if grid[r, c] == 8: is_template = True; break
            if not is_template:
                # Check if it contains 2 or 3
                colors = set(grid[r, c] for r, c in obj)
                if 2 in colors or 3 in colors:
                    marker_objs.append(obj)
                    
        for mobj in marker_objs:
            # Anchor: top-left of this marker object
            mr_min, mc_min = mobj.min(axis=0)
            m_color = grid[mr_min, mc_min]
            
            # Find corresponding anchor in template
            if m_color in markers_in_template:
                # Use the first one found (usually there's only one or they are symmetric)
                tr_m, tc_m = markers_in_template[m_color][0]
                
                # Stamp offset: template(tr_m, tc_m) aligns with grid(mr_min, mc_min)
                # So template(0, 0) aligns with (mr_min - tr_m, mc_min - tc_m)
                off_r, off_c = mr_min - tr_m, mc_min - tc_m
                
                for dr, dc, color in template_pixels:
                    nr, nc = off_r + dr, off_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        # Only overwrite if not already a marker?
                        # In ARC, usually we don't overwrite the marker itself if it's different.
                        # But here the template HAS the marker color.
                        output[nr, nc] = color
                        
        return output

    # Verify
    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    return [run_single(ti) for ti in solver.test_in]
