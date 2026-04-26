import numpy as np
from typing import List, Optional

def solve_diagonal_propagation_from_seeds_by_region_color(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies 'major' colors (regions) and 'seed' colors.
    Associates each major color with the seed color that is most adjacent to it.
    Propagates diagonal lines from each seed, recoloring only the pixels that 
    match the associated major color.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        
        # 1. Identify major colors (two most frequent non-zero)
        unique, counts = np.unique(grid, return_counts=True)
        color_counts = sorted([(int(u), int(c)) for u, c in zip(unique, counts) if u != 0], 
                              key=lambda x: x[1], reverse=True)
        
        if len(color_counts) < 2:
            return grid
            
        major_colors = {color_counts[0][0], color_counts[1][0]}
        
        # 2. Find seeds and build region mapping
        mapping = {}
        seeds = []
        
        for r in range(rows):
            for c in range(cols):
                val = int(grid[r, c])
                if val != 0 and val not in major_colors:
                    seeds.append((r, c, val))
                    
        for m_color in major_colors:
            adj_seed_counts = {}
            for r in range(rows):
                for c in range(cols):
                    if grid[r, c] == m_color:
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    v = int(grid[nr, nc])
                                    if v != 0 and v not in major_colors:
                                        adj_seed_counts[v] = adj_seed_counts.get(v, 0) + 1
            if adj_seed_counts:
                best_seed = max(adj_seed_counts, key=adj_seed_counts.get)
                mapping[m_color] = best_seed

        # 3. Propagate diagonals
        output = grid.copy()
        for sr, sc, s_color in seeds:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                curr_r, curr_c = sr + dr, sc + dc
                while 0 <= curr_r < rows and 0 <= curr_c < cols:
                    orig_val = int(grid[curr_r, curr_c])
                    if orig_val in mapping:
                        output[curr_r, curr_c] = mapping[orig_val]
                    curr_r += dr
                    curr_c += dc
        return output

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
