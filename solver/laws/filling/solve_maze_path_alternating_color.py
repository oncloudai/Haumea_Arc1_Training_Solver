
import numpy as np
from typing import List, Optional
from collections import deque

def solve_maze_path_alternating_color(solver) -> Optional[List[np.ndarray]]:
    for bg_color in range(10):
        consistent = True
        found_any = False
        for inp, out in solver.pairs:
            if inp.shape != out.shape: consistent = False; break
            
            unq, counts = np.unique(inp, return_counts=True)
            if len(unq) < 3: consistent = False; break
            
            non_bg = [c for c in unq if c != bg_color]
            if not non_bg: consistent = False; break
            
            maze_color = [c for c, cnt in sorted(zip(unq, counts), key=lambda x: x[1], reverse=True) if c != bg_color][0]
            endpoints = [c for c in unq if c != bg_color and c != maze_color]
            
            if not endpoints: consistent = False; break
            
            h, w = inp.shape
            ep_coords = []
            for ec in endpoints:
                for r, c in np.argwhere(inp == ec):
                    ep_coords.append((r, c, ec))
            
            if len(ep_coords) < 2: consistent = False; break
            
            def get_path(start, end):
                q = deque([start])
                parent = {start: None}
                while q:
                    curr = q.popleft()
                    if curr == end:
                        path = []
                        while curr is not None:
                            path.append(curr)
                            curr = parent[curr]
                        return path[::-1]
                    r, c = curr
                    for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in parent:
                            if inp[nr, nc] == maze_color or (nr, nc) == end:
                                parent[(nr, nc)] = (r, c)
                                q.append((nr, nc))
                return None

            res = inp.copy()
            path_found = False
            # Optimization: only check first pair for training match
            p1 = (ep_coords[0][0], ep_coords[0][1])
            p2 = (ep_coords[1][0], ep_coords[1][1])
            path = get_path(p1, p2)
            if path:
                c1, c2 = ep_coords[0][2], ep_coords[1][2]
                for idx, (pr, pc) in enumerate(path):
                    res[pr, pc] = c1 if idx % 2 == 0 else c2
                path_found = True
            
            if not path_found or not np.array_equal(res, out):
                # Try the other alternating order?
                if path_found:
                    res2 = inp.copy()
                    for idx, (pr, pc) in enumerate(path):
                        res2[pr, pc] = c2 if idx % 2 == 0 else c1
                    if np.array_equal(res2, out):
                        path_found = True
                        res = res2
                    else:
                        consistent = False; break
                else:
                    consistent = False; break
            found_any = True
            
        if consistent and found_any:
            results = []
            for ti in solver.test_in:
                h, w = ti.shape
                unq, counts = np.unique(ti, return_counts=True)
                # Ensure we have enough colors
                if len(unq) < 3: results.append(ti.copy()); continue
                
                maze_color_test = [c for c, cnt in sorted(zip(unq, counts), key=lambda x: x[1], reverse=True) if c != bg_color][0]
                endpoints_test = [c for c in unq if c != bg_color and c != maze_color_test]
                ep_coords_test = []
                for ec in endpoints_test:
                    for r, c in np.argwhere(ti == ec): ep_coords_test.append((r, c, ec))
                
                res_ti = ti.copy()
                if len(ep_coords_test) >= 2:
                    p1_t = (ep_coords_test[0][0], ep_coords_test[0][1])
                    p2_t = (ep_coords_test[1][0], ep_coords_test[1][1])
                    q = deque([p1_t]); parent = {p1_t: None}; path = None
                    while q:
                        curr = q.popleft()
                        if curr == p2_t:
                            path = []; c_p = curr
                            while c_p is not None: path.append(c_p); c_p = parent[c_p]
                            path = path[::-1]; break
                        r, c = curr
                        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in parent:
                                if ti[nr, nc] == maze_color_test or (nr, nc) == p2_t:
                                    parent[(nr, nc)] = (r, c); q.append((nr, nc))
                    if path:
                        c1, c2 = ep_coords_test[0][2], ep_coords_test[1][2]
                        # Determine order from training? 
                        # For now, just use c1, c2
                        for idx, (pr, pc) in enumerate(path): res_ti[pr, pc] = c1 if idx % 2 == 0 else c2
                results.append(res_ti)
            return results
    return None
