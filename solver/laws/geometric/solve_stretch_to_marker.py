import numpy as np
from typing import List, Optional

def solve_stretch_to_marker(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        for marker_color in range(1, 10):
            consistent = True
            found_any = False
            for inp, out in solver.pairs:
                m_coords = np.argwhere(inp == marker_color)
                if len(m_coords) != 1: consistent = False; break
                rm, cm = m_coords[0]
                
                obj_mask = (inp != bg) & (inp != marker_color)
                obj_coords = np.argwhere(obj_mask)
                if len(obj_coords) == 0: consistent = False; break
                r1, c1 = obj_coords.min(axis=0)
                r2, c2 = obj_coords.max(axis=0)
                
                pred = inp.copy(); pred[rm, cm] = bg
                worked = False
                if c1 <= cm <= c2: # Vertical
                    if rm > r2:
                        pred[r1:rm+1, c1:c2+1] = bg
                        pred[r1, c1:c2+1] = inp[r1, c1:c2+1]
                        pred[rm, c1:c2+1] = inp[r2, c1:c2+1]
                        for r in range(r1+1, rm):
                            pred[r, c1:c2+1] = inp[r1+1, c1:c2+1]
                        worked = True
                    elif rm < r1:
                        pred[rm:r2+1, c1:c2+1] = bg
                        pred[rm, c1:c2+1] = inp[r1, c1:c2+1]
                        pred[r2, c1:c2+1] = inp[r2, c1:c2+1]
                        for r in range(rm+1, r2):
                            pred[r, c1:c2+1] = inp[r1+1, c1:c2+1]
                        worked = True
                elif r1 <= rm <= r2: # Horizontal
                    if cm > c2:
                        pred[r1:r2+1, c1:cm+1] = bg
                        pred[r1:r2+1, c1] = inp[r1:r2+1, c1]
                        pred[r1:r2+1, cm] = inp[r1:r2+1, c2]
                        for c in range(c1+1, cm):
                            pred[r1:r2+1, c] = inp[r1:r2+1, c1+1]
                        worked = True
                    elif cm < c1:
                        pred[r1:r2+1, cm:c2+1] = bg
                        pred[r1:r2+1, cm] = inp[r1:r2+1, c1]
                        pred[r1:r2+1, c2] = inp[r1:r2+1, c2]
                        for c in range(cm+1, c2):
                            pred[r1:r2+1, c] = inp[r1:r2+1, c1+1]
                        worked = True
                
                if not worked or not np.array_equal(pred, out):
                    consistent = False; break
                found_any = True
            
            if consistent and found_any:
                def process(grid):
                    mc = np.argwhere(grid == marker_color)
                    if len(mc) != 1: return grid
                    rm, cm = mc[0]
                    om = (grid != bg) & (grid != marker_color)
                    oc = np.argwhere(om)
                    if len(oc) == 0: return grid
                    r1, c1 = oc.min(axis=0); r2, c2 = oc.max(axis=0)
                    res = grid.copy(); res[rm, cm] = bg
                    if c1 <= cm <= c2:
                        if rm > r2:
                            res[r1:rm+1, c1:c2+1] = bg; res[r1, c1:c2+1] = grid[r1, c1:c2+1]; res[rm, c1:c2+1] = grid[r2, c1:c2+1]
                            for r in range(r1+1, rm): res[r, c1:c2+1] = grid[r1+1, c1:c2+1]
                        elif rm < r1:
                            res[rm:r2+1, c1:c2+1] = bg; res[rm, c1:c2+1] = grid[r1, c1:c2+1]; res[r2, c1:c2+1] = grid[r2, c1:c2+1]
                            for r in range(rm+1, r2): res[r, c1:c2+1] = grid[r1+1, c1:c2+1]
                    elif r1 <= rm <= r2:
                        if cm > c2:
                            res[r1:r2+1, c1:cm+1] = bg; res[r1:r2+1, c1] = grid[r1:r2+1, c1]; res[r1:r2+1, cm] = grid[r1:r2+1, c2]
                            for c in range(c1+1, cm): res[r1:r2+1, c] = grid[r1:r2+1, c1+1]
                        elif cm < c1:
                            res[r1:r2+1, cm:c2+1] = bg; res[r1:r2+1, cm] = grid[r1:r2+1, c1]; res[r1:r2+1, c2] = grid[r1:r2+1, c2]
                            for c in range(cm+1, c2): res[r1:r2+1, c] = grid[r1:r2+1, c1+1]
                    return res
                return [process(ti) for ti in solver.test_in]
    return None
