import numpy as np
from typing import List, Optional

def solve_fit_grey_to_holes_v2(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies grey (5) objects.
    Finds an assignment of objects to holes (matching shapes in 0-space)
    that touches Red (2) and maximizes the total number of Red-adjacent pixels.
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

    def get_possible_fits(input_grid, gobj):
        rows, cols = input_grid.shape
        r_min, c_min = gobj.min(axis=0)
        rel_coords = gobj - [r_min, c_min]
        
        fits = []
        for r in range(rows):
            for c in range(cols):
                fit = True
                adj_count = 0
                for dr, dc in rel_coords:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols and input_grid[nr, nc] == 0):
                        fit = False; break
                    # Adjacency count
                    for nr2, nc2 in [(nr-1, nc), (nr+1, nc), (nr, nc-1), (nr, nc+1)]:
                        if 0 <= nr2 < rows and 0 <= nc2 < cols and input_grid[nr2, nc2] == 2:
                            adj_count += 1
                
                if fit and adj_count > 0:
                    fits.append({'pos': (r, c), 'adj': adj_count})
        return fits

    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        
        grey_objs = get_objects(grid, 5)
        if not grey_objs: return None
        
        all_fits = []
        for gobj in grey_objs:
            fits = get_possible_fits(grid, gobj)
            if not fits: return None
            all_fits.append(fits)
            
        best_assignment = [None] * len(grey_objs)
        max_total_adj = -1
        occupied = np.zeros((rows, cols), dtype=bool)
        
        current_assignment = [None] * len(grey_objs)
        
        def solve(idx, current_adj):
            nonlocal max_total_adj, best_assignment
            if idx == len(grey_objs):
                if current_adj > max_total_adj:
                    max_total_adj = current_adj
                    best_assignment = list(current_assignment)
                return
            
            # Pruning
            # potential = current_adj + sum(max(f['adj'] for f in all_fits[k]) for k in range(idx, len(grey_objs)))
            # if potential <= max_total_adj: return

            r_min_obj, c_min_obj = grey_objs[idx].min(axis=0)
            rel_coords = grey_objs[idx] - [r_min_obj, c_min_obj]
            
            for fit in all_fits[idx]:
                r, c = fit['pos']
                conflict = False
                for dr, dc in rel_coords:
                    if occupied[r + dr, c + dc]:
                        conflict = True; break
                if not conflict:
                    for dr, dc in rel_coords: occupied[r + dr, c + dc] = True
                    current_assignment[idx] = (r, c)
                    solve(idx + 1, current_adj + fit['adj'])
                    for dr, dc in rel_coords: occupied[r + dr, c + dc] = False
        
        solve(0, 0)
        
        if best_assignment[0] is not None:
            output = grid.copy()
            output[output == 5] = 0
            for i, pos in enumerate(best_assignment):
                if pos:
                    r, c = pos
                    r_min_obj, c_min_obj = grey_objs[i].min(axis=0)
                    rel_coords = grey_objs[i] - [r_min_obj, c_min_obj]
                    for dr, dc in rel_coords:
                        output[r + dr, c + dc] = 1
            return output
        return None

    # Verify
    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    return [run_single(ti) for ti in solver.test_in]
