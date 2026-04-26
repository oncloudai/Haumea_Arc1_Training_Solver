
import numpy as np
from typing import List, Optional

def solve_upscale_seed_to_fit_hollow_rect(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a hollow rectangle (container) and a seed object.
    Upscales the seed to fill the inner area of the container.
    Returns the container with the scaled seed inside.
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
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<rows and 0<=nc<cols and grid[nr,nc]!=bg and not visited[nr,nc]:
                                visited[nr,nc]=True; q.append((nr,nc))
                    objs.append(np.array(coords))
        return objs

    def is_hollow_rect(coords, grid):
        r_min, c_min = coords.min(axis=0); r_max, c_max = coords.max(axis=0)
        h, w = r_max - r_min + 1, c_max - c_min + 1
        if h < 3 or w < 3: return False, 0, 0, 0
        
        color = grid[r_min, c_min]
        # Check edges
        pixels = set(tuple(p) for p in coords)
        for r in range(r_min, r_max + 1):
            if (r, c_min) not in pixels or (r, c_max) not in pixels: return False, 0, 0, 0
            if grid[r, c_min] != color or grid[r, c_max] != color: return False, 0, 0, 0
        for c in range(c_min, c_max + 1):
            if (r_min, c) not in pixels or (r_max, c) not in pixels: return False, 0, 0, 0
            if grid[r_min, c] != color or grid[r_max, c] != color: return False, 0, 0, 0
            
        # Check inner is empty
        for r in range(r_min + 1, r_max):
            for c in range(c_min + 1, c_max):
                if (r, c) in pixels: return False, 0, 0, 0
                
        return True, h, w, color

    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        unique, counts = np.unique(grid, return_counts=True)
        bg = unique[np.argmax(counts)]
        
        objs = get_objects(grid, bg)
        if len(objs) < 2: return None
        
        container = None
        seed = None
        for i, obj in enumerate(objs):
            ok, h, w, color = is_hollow_rect(obj, grid)
            if ok:
                container = {'coords': obj, 'h': h, 'w': w, 'color': color}
                # Use other objects as potential seeds
                other_objs = [objs[j] for j in range(len(objs)) if i != j]
                # Combined other objects might form the seed
                seed_coords = np.concatenate(other_objs)
                sr_min, sc_min = seed_coords.min(axis=0)
                sr_max, sc_max = seed_coords.max(axis=0)
                sh, sw = sr_max - sr_min + 1, sc_max - sc_min + 1
                seed = {'coords': seed_coords, 'h': sh, 'w': sw, 'color': grid[seed_coords[0,0], seed_coords[0,1]], 'tl': (sr_min, sc_min)}
                break
        
        if not container or not seed: return None
        
        # Scale factor
        if (container['h'] - 2) % seed['h'] != 0 or (container['w'] - 2) % seed['w'] != 0:
            return None
        
        fh = (container['h'] - 2) // seed['h']
        fw = (container['w'] - 2) // seed['w']
        
        output = np.full((container['h'], container['w']), 0) # 0 is relative bg
        # Draw border
        output[0, :] = container['color']
        output[-1, :] = container['color']
        output[:, 0] = container['color']
        output[:, -1] = container['color']
        
        # Scale seed
        for r, c in seed['coords']:
            # Relative to seed TL
            dr, dc = r - seed['tl'][0], c - seed['tl'][1]
            color = grid[r, c]
            for rh in range(fh):
                for cw in range(fw):
                    output[1 + dr * fh + rh, 1 + dc * fw + cw] = color
                    
        return output

    # Verify
    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    return [run_single(ti) for ti in solver.test_in]
