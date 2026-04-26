import numpy as np
from typing import List, Optional

def solve_move_objects_into_containers(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies red (2) container (treated as one bounding box).
    Moves grey (5) objects into the container.
    Logic:
    1. Determine direction from Grey Center to Container Center.
    2. Move Grey to enter the Container from that direction.
    3. Skip the 'Wall' (rows/cols with Red pixels inside the bbox) and dock at the first 'Inside' (empty) row/col.
    4. Align centers on the non-movement axis.
    5. Flip the object across the movement axis.
    """
    def get_objects(grid, color):
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        objs = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == color and not visited[r, c]:
                    q = [(r, c)]; visited[r, c] = True
                    coords = []
                    while q:
                        cr, cc = q.pop(0)
                        coords.append((cr, cc))
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<rows and 0<=nc<cols and grid[nr,nc]==color and not visited[nr,nc]:
                                visited[nr,nc]=True; q.append((nr,nc))
                    objs.append(np.array(coords))
        return objs

    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        
        red_coords = np.argwhere(grid == 2)
        if len(red_coords) == 0: return None
        
        # Container BBox
        r_min, c_min = red_coords.min(axis=0)
        r_max, c_max = red_coords.max(axis=0)
        r_center_red = (r_min + r_max) / 2
        c_center_red = (c_min + c_max) / 2
        
        red_mask = np.zeros((rows, cols), dtype=bool)
        red_mask[red_coords[:,0], red_coords[:,1]] = True
        
        grey_objs = get_objects(grid, 5)
        if not grey_objs: return None
        
        output = grid.copy()
        # Clear original grey
        for gobj in grey_objs:
            for r, c in gobj: output[r, c] = 0
            
        for gobj in grey_objs:
            gr_center = (gobj.min(axis=0)[0] + gobj.max(axis=0)[0]) / 2
            gc_center = (gobj.min(axis=0)[1] + gobj.max(axis=0)[1]) / 2
            
            dr = gr_center - r_center_red
            dc = gc_center - c_center_red
            
            # Object BBox
            gr_min, gc_min = gobj.min(axis=0)
            gr_max, gc_max = gobj.max(axis=0)
            gh, gw = gr_max - gr_min + 1, gc_max - gc_min + 1
            
            # Mask
            g_mask = np.zeros((gh, gw), dtype=int)
            for r, c in gobj:
                g_mask[r - gr_min, c - gc_min] = 1
            
            # Target coords
            target_r, target_c = -1, -1
            flipped_mask = g_mask
            
            if abs(dr) > abs(dc):
                # Vertical movement
                # Align columns
                target_c = int(round(c_center_red - (gw - 1) / 2))
                flipped_mask = np.flipud(g_mask)
                
                # Find dock row
                if dr < 0: # From Top moving Down
                    curr = r_min
                    while curr <= r_max:
                        if np.any(red_mask[curr, c_min:c_max+1]):
                            curr += 1
                        else:
                            break
                    target_r = curr
                else: # From Bottom moving Up
                    curr = r_max
                    while curr >= r_min:
                        if np.any(red_mask[curr, c_min:c_max+1]):
                            curr -= 1
                        else:
                            break
                    target_r = curr - (gh - 1)
            else:
                # Horizontal movement
                # Align rows
                target_r = int(round(r_center_red - (gh - 1) / 2))
                flipped_mask = np.fliplr(g_mask)
                
                # Find dock col
                if dc < 0: # From Left moving Right
                    curr = c_min
                    while curr <= c_max:
                        if np.any(red_mask[r_min:r_max+1, curr]):
                            curr += 1
                        else:
                            break
                    target_c = curr
                else: # From Right moving Left
                    curr = c_max
                    while curr >= c_min:
                        if np.any(red_mask[r_min:r_max+1, curr]):
                            curr -= 1
                        else:
                            break
                    target_c = curr - (gw - 1)
            
            # Place
            for r in range(gh):
                for c in range(gw):
                    if flipped_mask[r, c] == 1:
                        nr, nc = target_r + r, target_c + c
                        if 0 <= nr < rows and 0 <= nc < cols:
                            output[nr, nc] = 5
                            
        return output

    # Pairs check
    for inp, out_exp in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out_exp): return None
        
    return [run_single(ti) for ti in solver.test_in]
