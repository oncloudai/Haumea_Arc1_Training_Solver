import numpy as np
from typing import List, Optional

def solve_propagate_seeds_with_deflection(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies 'seeds' (color 8) and 'deflectors' (color 2).
    Propagates seeds in a primary direction (vertical or horizontal) until a 
    deflector is encountered. At each deflector, the seeds shift their 
    secondary coordinate away from the deflector.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        out = grid.copy()
        
        seed_color = 8
        deflector_color = 2
        
        s_rows, s_cols = np.where(grid == seed_color)
        d_rows, d_cols = np.where(grid == deflector_color)
        
        if len(s_rows) == 0 or len(d_rows) == 0:
            return None
            
        # Determine propagation type
        is_vertical = len(set(d_cols)) == 1
        is_horizontal = len(set(d_rows)) == 1
        
        if is_vertical:
            cd = d_cols[0]
            sr = s_rows[0]
            v_dir = 1 if sr == 0 else -1
            
            P = set(s_cols[s_rows == sr])
            R_def = sorted(list(set(d_rows)), reverse=(v_dir == -1))
            
            curr_r = sr
            target_rows = R_def + ([h] if v_dir == 1 else [-1])
            
            for rd in target_rows:
                r_range = range(curr_r, rd, v_dir)
                for r in r_range:
                    for p in P:
                        if 0 <= p < w:
                            out[r, p] = seed_color
                
                if rd == h or rd == -1: break
                
                # Shift at row rd
                p_ref = next(iter(P))
                shift = 1 if p_ref > cd else -1
                
                new_P = set()
                for p in P:
                    np_pos = p + shift
                    # Blocked by deflector column
                    if 0 <= np_pos < w and np_pos != cd:
                        new_P.add(np_pos)
                P = new_P
                curr_r = rd
                
        elif is_horizontal:
            rd = d_rows[0]
            sc = s_cols[0]
            h_dir = 1 if sc == 0 else -1
            
            P = set(s_rows[s_cols == sc])
            C_def = sorted(list(set(d_cols)), reverse=(h_dir == -1))
            
            curr_c = sc
            target_cols = C_def + ([w] if h_dir == 1 else [-1])
            
            for cd in target_cols:
                c_range = range(curr_c, cd, h_dir)
                for c in c_range:
                    for p in P:
                        if 0 <= p < h:
                            out[p, c] = seed_color
                
                if cd == w or cd == -1: break
                
                p_ref = next(iter(P))
                shift = 1 if p_ref > rd else -1
                
                new_P = set()
                for p in P:
                    np_pos = p + shift
                    if 0 <= np_pos < h and np_pos != rd:
                        new_P.add(np_pos)
                P = new_P
                curr_c = cd
        else:
            return None
            
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
