
import numpy as np
from typing import List, Optional

def solve_combine_two_objects_3x3(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies two non-zero objects in the grid.
    Finds a relative offset between them such that their union fits in a 3x3 window
    and they have NO overlapping non-zero pixels.
    Outputs the resulting 3x3 block.
    """
    def get_objects(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        objs = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0 and not visited[r, c]:
                    color = grid[r, c]
                    q = [(r, c)]; visited[r, c] = True
                    coords = []
                    while q:
                        cr, cc = q.pop(0)
                        coords.append((cr, cc))
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<rows and 0<=nc<cols and grid[nr,nc]!=0 and not visited[nr,nc]:
                                visited[nr,nc]=True; q.append((nr,nc))
                    # All pixels of this object (can be multiple colors, but flood fill handles it)
                    objs.append(coords)
        return objs

    def run_single(input_grid):
        grid = np.array(input_grid)
        objs_coords = get_objects(grid)
        if len(objs_coords) != 2: return None
        
        # Extract the two objects as (relative_coord, color) lists
        o1 = [(tuple(np.array(p) - np.array(objs_coords[0]).min(axis=0)), grid[p]) for p in objs_coords[0]]
        o2 = [(tuple(np.array(p) - np.array(objs_coords[1]).min(axis=0)), grid[p]) for p in objs_coords[1]]
        
        # Try all relative offsets (dr, dc) of o2 with respect to o1
        # Range of offsets: o1 can be anywhere in 3x3, o2 can be anywhere in 3x3.
        # So dr in [-2, 2], dc in [-2, 2]? No, relative to o1's bounding box top-left.
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                # Potential union
                union_pixels = {}
                conflict = False
                
                # Add o1
                for p, color in o1:
                    union_pixels[p] = color
                
                # Add o2 shifted by (dr, dc)
                for p, color in o2:
                    np_shifted = (p[0] + dr, p[1] + dc)
                    if np_shifted in union_pixels:
                        conflict = True; break
                    union_pixels[np_shifted] = color
                
                if conflict: continue
                
                # Check if union fits in a 3x3
                u_coords = np.array(list(union_pixels.keys()))
                r_min, c_min = u_coords.min(axis=0)
                r_max, c_max = u_coords.max(axis=0)
                if r_max - r_min < 3 and c_max - c_min < 3:
                    # Fits! Construct 3x3.
                    res = np.zeros((3, 3), dtype=int)
                    for (pr, pc), color in union_pixels.items():
                        res[pr - r_min, pc - c_min] = color
                    yield res

    # Verify on pairs
    # Note: run_single is a generator because there might be multiple valid 3x3s.
    # We need to find one that consistently works for all pairs.
    # But since objects are unique, usually there is only one.
    
    # Actually, the problem is that we don't know the exact offset.
    # Let's just try to find ANY working 3x3 for each pair.
    
    consistent = True
    for inp, out_expected in solver.pairs:
        found_match = False
        for pred in run_single(inp):
            if np.array_equal(pred, out_expected):
                found_match = True; break
        if not found_match:
            consistent = False; break
            
    if consistent:
        test_preds = []
        for inp in solver.test_in:
            preds = list(run_single(inp))
            if preds:
                test_preds.append(preds[0]) # Pick first valid
            else:
                test_preds.append(np.zeros((3, 3), dtype=int))
        return test_preds
        
    return None
