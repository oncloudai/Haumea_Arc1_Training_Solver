import numpy as np
from typing import List, Optional

def solve_recolor_by_orthogonal_adjacency_rule(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies rules of the form: 
    If pixel of color A is orthogonally adjacent to color B, change pixel to color C.
    If pixel of color B is orthogonally adjacent to color A, change pixel to color D.
    Infors the colors A, B, C, D from the first training pair.
    """
    def get_rules(inp, out):
        rules = [] # List of (color_source, color_adj, color_target)
        rows, cols = inp.shape
        for r in range(rows):
            for c in range(cols):
                if inp[r, c] != out[r, c]:
                    # Changed! Check neighbors in input
                    source = int(inp[r, c])
                    target = int(out[r, c])
                    adj_colors = set()
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            adj_colors.add(int(inp[nr, nc]))
                    
                    if 0 in adj_colors: adj_colors.remove(0)
                    if source in adj_colors: adj_colors.remove(source)
                    
                    for adj in adj_colors:
                        rules.append((source, adj, target))
        return list(set(rules))

    if not solver.pairs: return None
    rules = get_rules(solver.train_in[0], solver.train_out[0])
    if not rules: return None
    
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        out = grid.copy()
        # Apply all rules simultaneously (using original grid for adjacency check)
        for r in range(rows):
            for c in range(cols):
                source = grid[r, c]
                for r_src, r_adj, r_tgt in rules:
                    if source == r_src:
                        # Check adjacency to r_adj in input grid
                        has_adj = False
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if grid[nr, nc] == r_adj:
                                    has_adj = True; break
                        if has_adj:
                            out[r, c] = r_tgt
                            break
        return out

    # Verify on all pairs
    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
