import numpy as np
from typing import List, Optional

def solve_replicate_red_seed_at_matched_local_envs(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a connected component of red (color 2) as a 'seed'.
    Identifies its local environment (bounding box colors).
    Searches the grid for all other locations matching that local environment 
    and replicates the red seed there.
    """
    def get_symmetries(grid):
        soms = []
        curr = np.array(grid)
        for _ in range(4):
            soms.append(curr.copy())
            soms.append(np.fliplr(curr).copy())
            curr = np.rot90(curr)
        unique_soms = []
        seen = set()
        for s in soms:
            t = (s.shape, tuple(s.flatten()))
            if t not in seen:
                unique_soms.append(s)
                seen.add(t)
        return unique_soms

    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        output_grid = grid.copy()
        
        # 1. Identify connected components of red
        visited = np.zeros(grid.shape, dtype=bool)
        red_mask = (grid == 2)
        seeds = []
        for r in range(h):
            for c in range(w):
                if red_mask[r, c] and not visited[r, c]:
                    pixels = []
                    q = [(r, c)]
                    visited[r, c] = True
                    while q:
                        curr_r, curr_c = q.pop(0)
                        pixels.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < h and 0 <= nc < w and red_mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                    pixels = np.array(pixels)
                    rm, cm = pixels.min(axis=0); rX, cX = pixels.max(axis=0)
                    shape_h, shape_w = rX - rm + 1, cX - cm + 1
                    red_t = np.zeros((shape_h, shape_w), dtype=int)
                    for pr, pc in pixels:
                        red_t[pr - rm, pc - cm] = 2
                    env_t = grid[rm:rX+1, cm:cX+1].copy()
                    seeds.append({'red': red_t, 'env': env_t})
                    
        if not seeds: return grid
        
        # Take the first seed found (as per Arc logic usually one rule applies)
        seed = seeds[0]
        red_syms = get_symmetries(seed['red'])
        env_syms = get_symmetries(seed['env'])
        
        for es, rs in zip(env_syms, red_syms):
            th, tw = es.shape
            match_mask = (es != 2)
            match_vals = es[match_mask]
            
            for r in range(h - th + 1):
                for c in range(w - tw + 1):
                    local_vals = grid[r:r+th, c:c+tw][match_mask]
                    if np.array_equal(local_vals, match_vals):
                        for tr in range(th):
                            for tc in range(tw):
                                if rs[tr, tc] == 2:
                                    output_grid[r + tr, c + tc] = 2
        return output_grid

    for inp, out in solver.pairs:
        if not np.array_equal(apply_logic(inp), out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
