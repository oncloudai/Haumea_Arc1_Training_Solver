import numpy as np
from typing import List, Optional

def solve_marker_seed_copy(solver) -> Optional[List[np.ndarray]]:
    """
    Red (2) markers trigger horizontal flip copy of their source cluster.
    Green (3) markers trigger identity copy of their source cluster.
    Source cluster is the one containing the marker and other non-zero pixels.
    Target markers are isolated.
    """
    def get_clusters(grid):
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        clusters = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0 and not visited[r, c]:
                    cluster = []
                    q = [(r, c)]
                    visited[r, c] = True
                    while q:
                        curr_r, curr_c = q.pop(0)
                        cluster.append((curr_r, curr_c))
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0: continue
                                nr, nc = curr_r + dr, curr_c + dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    if grid[nr, nc] != 0 and not visited[nr, nc]:
                                        visited[nr, nc] = True
                                        q.append((nr, nc))
                    clusters.append(cluster)
        return clusters

    def run_single(input_grid):
        input_grid = np.array(input_grid)
        output_grid = input_grid.copy()
        rows, cols = input_grid.shape
        clusters = get_clusters(input_grid)
        
        marker_pixels = []
        for r in range(rows):
            for c in range(cols):
                if input_grid[r, c] in [2, 3]:
                    marker_pixels.append((r, c, input_grid[r, c]))
                    
        if not marker_pixels:
            return None

        sources = []
        targets = []
        for r, c, color in marker_pixels:
            is_isolated = True
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if input_grid[nr, nc] != 0:
                            is_isolated = False
                            break
                if not is_isolated: break
            
            if is_isolated:
                targets.append((r, c, color))
            else:
                sources.append((r, c, color))
                
        if not sources or not targets:
            return None

        templates = {}
        for r, c, color in sources:
            source_cluster = None
            for cluster in clusters:
                if (r, c) in cluster:
                    source_cluster = cluster
                    break
            if source_cluster:
                template_pixels = []
                for pr, pc in source_cluster:
                    if (pr, pc) == (r, c): continue
                    template_pixels.append((pr - r, pc - c, input_grid[pr, pc]))
                templates[color] = template_pixels
                
        if not templates:
            return None

        changed = False
        for tr, tc, color in targets:
            if color in templates:
                template_pixels = templates[color]
                for dr, dc, p_color in template_pixels:
                    if color == 2: # Red -> flip_h
                        tr_r, tr_c = dr, -dc
                    elif color == 3: # Green -> identity
                        tr_r, tr_c = dr, dc
                    else:
                        continue
                    
                    nr, nc = tr + tr_r, tc + tr_c
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if output_grid[nr, nc] != p_color:
                            output_grid[nr, nc] = p_color
                            changed = True
                            
        return output_grid if changed else None

    # Verify on training pairs
    for inp, out in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out):
            return None
            
    # Generate test predictions
    test_preds = []
    for inp in solver.test_in:
        pred = run_single(inp)
        if pred is None:
            # If no change on test, but correct on train, maybe still okay if train HAD changes.
            # But usually test should also have changes.
            test_preds.append(inp.copy())
        else:
            test_preds.append(pred)
            
    return test_preds
