import numpy as np
from typing import List, Optional

def solve_fill_cross_corridors_between_clusters(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies sparse vertical and horizontal 'corridor' bands that separate dense clusters of non-zero pixels.
    Fills the cross-shaped intersection and extensions of these corridors with color 3.
    1. Finds vertical and horizontal corridors using non-zero pixel density.
    2. Erodes these bands to find core fill regions.
    3. Extends the fill based on cluster locations relative to corridors.
    """
    def erode_1d(mask, border_value=True):
        n = len(mask)
        result = mask.copy()
        for i in range(n):
            if mask[i]:
                left  = mask[i - 1] if i > 0     else border_value
                right = mask[i + 1] if i < n - 1 else border_value
                if not left or not right:
                    result[i] = False
        return result

    def find_groups(bool_array):
        groups, start = [], None
        for i, v in enumerate(bool_array):
            if v and start is None:
                start = i
            elif not v and start is not None:
                groups.append((start, i - 1))
                start = None
        if start is not None:
            groups.append((start, len(bool_array) - 1))
        return groups

    def erode_group(start, end):
        return list(range(start + 1, end)) if end > start else [start]

    def find_corridor_1d(nz_arr):
        sorted_u = np.sort(np.unique(nz_arr))
        threshold = 0
        if len(sorted_u) > 1:
            gaps = np.diff(sorted_u)
            if np.max(gaps) > 3:
                threshold = sorted_u[np.argmax(gaps)]
        sparse = nz_arr <= threshold
        return sparse, erode_1d(sparse)

    def apply_logic(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        fill_mask = np.zeros((rows, cols), dtype=bool)

        # Vertical corridor
        col_nz = np.array([np.sum(grid[:, c] != 0) for c in range(cols)])
        v_sparse, v_corr = find_corridor_1d(col_nz)

        if np.any(v_corr):
            v_idx = np.where(v_sparse)[0]
            v_left_bnd = int(v_idx[0]) - 1
            v_right_bnd = int(v_idx[-1]) + 1
            v_cs = int(np.where(v_corr)[0][0])
            v_ce = int(np.where(v_corr)[0][-1])

            band_zero = np.array([
                np.all(grid[r, v_idx[0]:v_idx[-1] + 1] == 0) for r in range(rows)
            ])
            for r in np.where(erode_1d(band_zero))[0]:
                fill_mask[r, v_cs:v_ce + 1] = True

            row_all_left = np.zeros(rows, dtype=bool)
            row_all_right = np.zeros(rows, dtype=bool)
            for r in range(rows):
                nz = np.where(grid[r] != 0)[0]
                if not len(nz): continue
                if np.all(nz <= v_left_bnd): row_all_left[r] = True
                if np.all(nz >= v_right_bnd): row_all_right[r] = True

            for s, e in find_groups(row_all_left):
                for r in erode_group(s, e):
                    fill_mask[r, v_cs:] = True

            for s, e in find_groups(row_all_right):
                for r in erode_group(s, e):
                    fill_mask[r, :v_ce + 1] = True

        # Horizontal corridor
        row_nz = np.array([np.sum(grid[r, :] != 0) for r in range(rows)])
        h_sparse, h_corr = find_corridor_1d(row_nz)

        if np.any(h_corr):
            h_idx = np.where(h_sparse)[0]
            h_top_bnd = int(h_idx[0]) - 1
            h_bot_bnd = int(h_idx[-1]) + 1
            h_rs = int(np.where(h_corr)[0][0])
            h_re = int(np.where(h_corr)[0][-1])

            band_zero = np.array([
                np.all(grid[h_idx[0]:h_idx[-1] + 1, c] == 0) for c in range(cols)
            ])
            for c in np.where(erode_1d(band_zero))[0]:
                fill_mask[h_rs:h_re + 1, c] = True

            col_all_top = np.zeros(cols, dtype=bool)
            col_all_bot = np.zeros(cols, dtype=bool)
            for c in range(cols):
                nz = np.where(grid[:, c] != 0)[0]
                if not len(nz): continue
                if np.all(nz <= h_top_bnd): col_all_top[c] = True
                if np.all(nz >= h_bot_bnd): col_all_bot[c] = True

            for s, e in find_groups(col_all_top):
                for c in erode_group(s, e):
                    fill_mask[h_rs:, c] = True

            for s, e in find_groups(col_all_bot):
                for c in erode_group(s, e):
                    fill_mask[:h_re + 1, c] = True

        if not np.any(fill_mask): return None
        
        output = grid.copy()
        output[fill_mask & (grid == 0)] = 3
        return output

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
