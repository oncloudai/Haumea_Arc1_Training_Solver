
import numpy as np
from typing import List, Optional

def solve_marker_scaling_stamping(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies marker objects (m1_color and m2_color).
    Finds a template where markers are connected by conn_color.
    Stamps scaled and transformed versions of the connector pixels at ALL valid marker pairs.
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

    def transform_point(r, c, t_idx):
        if t_idx == 0: return r, c
        if t_idx == 1: return c, -r
        if t_idx == 2: return -r, -c
        if t_idx == 3: return -c, r
        if t_idx == 4: return r, -c
        if t_idx == 5: return -r, c
        if t_idx == 6: return c, r
        if t_idx == 7: return -c, -r
        return r, c

    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        unique, counts = np.unique(grid, return_counts=True)
        bg = unique[np.argmax(counts)]
        
        objs = get_objects(grid, bg)
        template = None
        m1_color, m2_color, conn_color = -1, -1, -1
        
        for obj in objs:
            obj_colors = set(grid[pr, pc] for pr, pc in obj)
            if len(obj_colors) >= 3:
                for cand_conn_color in obj_colors:
                    is_marker = False
                    for other_obj in objs:
                        # Check if other_obj is a marker of color cand_conn_color
                        if len(set(grid[pr, pc] for pr, pc in other_obj)) == 1 and grid[other_obj[0,0], other_obj[0,1]] == cand_conn_color:
                            # Is it disjoint from the template?
                            is_disjoint = True
                            for p1 in obj:
                                for p2 in other_obj:
                                    if np.array_equal(p1, p2): is_disjoint = False; break
                                if not is_disjoint: break
                            if is_disjoint:
                                is_marker = True; break
                    if not is_marker:
                        conn_color = cand_conn_color
                        markers = [cc for cc in obj_colors if cc != conn_color]
                        if len(markers) == 2:
                            m1_color, m2_color = markers
                            template = obj; break
                if template is not None: break
        
        if template is None: return None
        
        m1_src_coords = [tuple(p) for p in template if grid[p[0], p[1]] == m1_color]
        m2_src_coords = [tuple(p) for p in template if grid[p[0], p[1]] == m2_color]
        conn_src_coords = [tuple(p) for p in template if grid[p[0], p[1]] == conn_color]
        
        m1_src_tl = np.array(m1_src_coords).min(axis=0)
        m2_src_tl = np.array(m2_src_coords).min(axis=0)
        v_src = m2_src_tl - m1_src_tl
        
        rel_conn = [(r - m1_src_tl[0], c - m1_src_tl[1]) for r, c in conn_src_coords]
        
        all_m1 = [o for o in objs if len(set(grid[pr, pc] for pr, pc in o)) == 1 and grid[o[0,0], o[0,1]] == m1_color]
        all_m2 = [o for o in objs if len(set(grid[pr, pc] for pr, pc in o)) == 1 and grid[o[0,0], o[0,1]] == m2_color]
        
        output = grid.copy()
        found_any = False
        
        for m1_obj in all_m1:
            m1_tl = m1_obj.min(axis=0)
            for m2_obj in all_m2:
                m2_tl = m2_obj.min(axis=0)
                v_new = m2_tl - m1_tl
                
                for t_idx in range(8):
                    tr, tc = transform_point(v_src[0], v_src[1], t_idx)
                    fr, fc = -1, -1
                    if tr != 0:
                        if v_new[0] % tr == 0 and v_new[0] // tr > 0: fr = v_new[0] // tr
                    elif v_new[0] == 0: fr = 1
                    
                    if tc != 0:
                        if v_new[1] % tc == 0 and v_new[1] // tc > 0: fc = v_new[1] // tc
                    elif v_new[1] == 0: fc = 1
                    
                    if fr > 0 and fc > 0:
                        # VALID PAIR!
                        for dr, dc in rel_conn:
                            tr_c, tc_c = transform_point(dr, dc, t_idx)
                            for r_step in range(fr):
                                for c_step in range(fc):
                                    nr, nc = m1_tl[0] + tr_c * fr + r_step, m1_tl[1] + tc_c * fc + c_step
                                    if 0 <= nr < rows and 0 <= nc < cols:
                                        if output[nr, nc] == bg:
                                            output[nr, nc] = conn_color
                                            found_any = True
                        break
                        
        return output if found_any else None

    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    return [run_single(ti) for ti in solver.test_in]
