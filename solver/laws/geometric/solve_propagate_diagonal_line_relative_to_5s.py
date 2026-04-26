import numpy as np
from typing import List, Optional

def solve_propagate_diagonal_line_relative_to_5s(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a main diagonal line of a specific color.
    Identifies components of color 5.
    Calculates a new diagonal position based on the size and position of each color 5 component
    relative to the main diagonal, and draws the main color there.
    """
    def label_components(mask):
        rows, cols = mask.shape
        labeled = np.zeros((rows, cols), dtype=int)
        count = 0
        for r in range(rows):
            for c in range(cols):
                if mask[r, c] and labeled[r, c] == 0:
                    count += 1
                    q = [(r, c)]
                    labeled[r, c] = count
                    while q:
                        curr_r, curr_c = q.pop(0)
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and labeled[nr, nc] == 0:
                                labeled[nr, nc] = count
                                q.append((nr, nc))
        return labeled, count

    def apply_logic(input_grid):
        input_grid = np.array(input_grid)
        rows, cols = input_grid.shape
        unique = np.unique(input_grid)
        main_colors = [c for c in unique if c != 0 and c != 5]
        if not main_colors:
            return input_grid
        main_color = main_colors[0]
        
        in_main = np.argwhere(input_grid == main_color)
        if len(in_main) == 0:
            return input_grid
        
        ks_in = [p[0] - p[1] for p in in_main]
        k_in = int(np.round(np.mean(ks_in)))
        
        c5_mask = (input_grid == 5)
        labeled_c5, num_c5 = label_components(c5_mask)
        
        output_grid = input_grid.copy()
        output_grid[input_grid == 5] = 0
        
        for i in range(1, num_c5 + 1):
            comp_pixels = np.argwhere(labeled_c5 == i)
            ks = [p[0] - p[1] for p in comp_pixels]
            min_k, max_k = min(ks), max(ks)
            width = max_k - min_k + 1
            mean_k = np.mean(ks)
            
            sign = 1 if mean_k > k_in else -1
            k_new = k_in + sign * (width + 2)
            
            for r in range(rows):
                c = r - k_new
                if 0 <= c < cols:
                    output_grid[r, c] = int(main_color)
                    
        return output_grid

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if not np.array_equal(pred, out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
