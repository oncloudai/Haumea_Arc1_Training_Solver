
import numpy as np
from typing import List, Optional

def solve_scaled_stamping_by_marker_size(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a source marker set (red color 2) and a source object (color 1).
    Calculates scaling factors for other red marker sets based on marker size.
    Stamps scaled versions of the source object relative to those markers.
    """
    def get_red_objects(grid):
        grid = np.array(grid)
        labeled = np.zeros_like(grid); curr = 1
        coords = np.argwhere(grid == 2)
        for r, c in coords:
            if labeled[r, c] == 0:
                q = [(r, c)]; labeled[r, c] = curr
                while q:
                    cr, cc = q.pop(0)
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<grid.shape[0] and 0<=nc<grid.shape[1] and grid[nr,nc] == 2 and labeled[nr,nc] == 0:
                                labeled[nr,nc] = curr; q.append((nr,nc))
                curr += 1
        objs = []
        for i in range(1, curr):
            c = np.argwhere(labeled == i)
            objs.append((c.min(axis=0), c.max(axis=0)))
        return objs

    def apply_logic(grid):
        grid = np.array(grid)
        out = grid.copy()
        red_objs = get_red_objects(grid)
        blue_coords = np.argwhere(grid == 1)
        if len(blue_coords) == 0: return grid
        
        r1_b, c1_b = blue_coords.min(axis=0); r2_b, c2_b = blue_coords.max(axis=0)
        source_indices = []
        for idx, (mi, ma) in enumerate(red_objs):
            dist = max(0, mi[0] - r2_b, r1_b - ma[0]) + max(0, mi[1] - c2_b, c1_b - ma[1])
            if dist <= 1: source_indices.append(idx)
        
        if not source_indices: return grid
        
        source_objs = [red_objs[i] for i in source_indices]
        s_mi = np.min([o[0] for o in source_objs], axis=0)
        source_marker_size = source_objs[0][1][0] - source_objs[0][0][0] + 1
        source_rel = [(r - s_mi[0], c - s_mi[1]) for r, c in blue_coords]
        
        other_indices = [idx for idx in range(len(red_objs)) if idx not in source_indices]
        
        cols = {}
        for idx in other_indices:
            c = red_objs[idx][0][1]
            if c not in cols: cols[c] = []
            cols[c].append(idx)
        
        any_pair = False
        for c in cols:
            indices = sorted(cols[c], key=lambda x: red_objs[x][0][0])
            for i in range(0, len(indices), 2):
                if i + 1 < len(indices):
                    idx1 = indices[i]; mi1, ma1 = red_objs[idx1]
                    target_marker_size = ma1[0] - mi1[0] + 1
                    S = target_marker_size // source_marker_size
                    for dr, dc in source_rel:
                        for br in range(S):
                            for bc in range(S):
                                nr, nc = mi1[0] + dr * S + br, mi1[1] + dc * S + bc
                                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]: out[nr, nc] = 1
                    any_pair = True
                    
        rows = {}
        for idx in other_indices:
            r = red_objs[idx][0][0]
            if r not in rows: rows[r] = []
            rows[r].append(idx)
        
        for r in rows:
            indices = sorted(rows[r], key=lambda x: red_objs[x][0][1])
            for i in range(0, len(indices), 2):
                if i + 1 < len(indices):
                    idx1 = indices[i]; mi1, ma1 = red_objs[idx1]
                    target_marker_size = ma1[0] - mi1[0] + 1
                    S = target_marker_size // source_marker_size
                    for dr, dc in source_rel:
                        for br in range(S):
                            for bc in range(S):
                                nr, nc = mi1[0] + dr * S + br, mi1[1] + dc * S + bc
                                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]: out[nr, nc] = 1
                    any_pair = True

        if not any_pair:
            for idx in other_indices:
                mi, ma = red_objs[idx]
                target_marker_size = ma[0] - mi[0] + 1
                S = target_marker_size // source_marker_size
                for dr, dc in source_rel:
                    for br in range(S):
                        for bc in range(S):
                            nr, nc = mi[0] + dr * S + br, mi[1] + dc * S + bc
                            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]: out[nr, nc] = 1
        return out

    for inp, out in solver.pairs:
        if not np.array_equal(apply_logic(inp), out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
