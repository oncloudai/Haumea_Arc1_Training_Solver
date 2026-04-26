import numpy as np
from typing import List, Optional
from collections import Counter

def solve_diagonal_ray_expansion(solver) -> Optional[List[np.ndarray]]:
    consistent = True; found_any = False
    for inp, out in solver.pairs:
        h, w = inp.shape
        counts = Counter(inp.flatten())
        seeds = [c for c, count in counts.items() if count == 1]
        if not seeds: consistent = False; break
        
        pred = inp.copy()
        for s_color in seeds:
            r0, c0 = np.argwhere(inp == s_color)[0]
            # Find majority neighbor color (excluding itself)
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < h and 0 <= nc < w:
                    neighbors.append(inp[nr, nc])
            if not neighbors: continue
            bg_color = Counter(neighbors).most_common(1)[0][0]
            
            # BFS for diagonal expansion
            queue = [(r0, c0)]
            visited = set([(r0, c0)])
            while queue:
                r, c = queue.pop(0)
                pred[r, c] = s_color
                found_any = True
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                        if inp[nr, nc] == bg_color:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                            
        if not np.array_equal(pred, out):
            consistent = False; break
            
    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            h, w = ti.shape; res = ti.copy()
            counts = Counter(ti.flatten())
            seeds = [c for c, count in counts.items() if count == 1]
            for s_color in seeds:
                r0, c0 = np.argwhere(ti == s_color)[0]
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r0 + dr, c0 + dc
                    if 0 <= nr < h and 0 <= nc < w: neighbors.append(ti[nr, nc])
                if not neighbors: continue
                bg_color = Counter(neighbors).most_common(1)[0][0]
                queue = [(r0, c0)]; visited = set([(r0, c0)])
                while queue:
                    r, c = queue.pop(0); res[r, c] = s_color
                    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                            if ti[nr, nc] == bg_color:
                                visited.add((nr, nc)); queue.append((nr, nc))
            results.append(res)
        return results
    return None
