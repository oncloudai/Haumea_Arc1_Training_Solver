import numpy as np
from typing import List, Optional

def solve_replicate_shape_in_matching_quadrant_pockets(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a source shape of a specific color (usually red 2).
    Normalizes the shape and finds potential placement locations in empty regions (0).
    Placement criteria:
    1. Must be all 0s in the input.
    2. Must be within the same quadrant (split by mid_r, mid_c).
    3. If the shape's centroid is not in the shape, the implied center must be a specific marker color (usually 5).
    """
    def normalize_shape(cells):
        cells = list(cells)
        min_r = min(r for r, c in cells)
        min_c = min(c for r, c in cells)
        return frozenset((r - min_r, c - min_c) for r, c in cells)

    def apply_logic(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        mid_r, mid_c = rows // 2, cols // 2

        # Identify source color (non-zero, non-marker)
        unique_colors = np.unique(grid)
        # We assume 2 is the shape color and 5 is the marker based on the specific task,
        # but let's try to infer if possible. 
        # For this law, we'll use 2 and 5 as defaults or infer from first pair.
        shape_color = 2
        marker_color = 5

        red_cells = frozenset(tuple(p) for p in np.argwhere(grid == shape_color))
        if not red_cells:
            return grid

        red_shape = normalize_shape(red_cells)
        tcells = list(red_shape)
        max_dr = max(r for r, c in tcells)
        max_dc = max(c for r, c in tcells)

        centroid_r = sum(r for r, c in tcells) / len(tcells)
        centroid_c = sum(c for r, c in tcells) / len(tcells)
        center_rel = (round(centroid_r), round(centroid_c))
        center_in_shape = center_rel in red_shape

        candidates = []
        for r0 in range(rows - max_dr):
            for c0 in range(cols - max_dc):
                cells = frozenset((r0 + dr, c0 + dc) for dr, dc in tcells)
                if cells == red_cells: continue
                if not all(grid[r, c] == 0 for r, c in cells): continue
                
                quads = set((r // mid_r, c // mid_c) for r, c in cells)
                if len(quads) > 1: continue

                if not center_in_shape:
                    center_abs = (r0 + center_rel[0], c0 + center_rel[1])
                    if not (0 <= center_abs[0] < rows and 0 <= center_abs[1] < cols): continue
                    if grid[center_abs[0], center_abs[1]] != marker_color: continue

                candidates.append(cells)

        output = grid.copy()
        placed = set()
        for cells in sorted(candidates, key=lambda x: sorted(x)):
            if cells & placed: continue
            for r, c in cells:
                output[r, c] = shape_color
            placed |= cells

        return output

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
