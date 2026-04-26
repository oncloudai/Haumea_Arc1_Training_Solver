import numpy as np
from typing import List, Optional

def solve_checkerboard_flood_fill_around_obstacles(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies non-zero seed colors (excluding a potential 'obstacle' color).
    Determines a checkerboard mapping (r+c)%2 -> color from existing seeds.
    Flood fills background pixels (0) reachable from seeds using the checkerboard pattern.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        
        # In task b782dc8a, obstacle color is 8.
        # Let's find colors that are not background (0).
        unique = np.unique(grid)
        non_bg = [int(c) for c in unique if c != 0]
        
        if not non_bg: return grid
        
        # Heuristic: the 'obstacle' color is often 8 or 5. 
        # For b782dc8a, it's 8. Let's see if 8 is present.
        obstacle_color = 8 if 8 in non_bg else None
        seed_colors = [c for c in non_bg if c != obstacle_color]
        
        if not seed_colors: return grid
        
        # Determine phase mapping: (r+c)%2 -> color
        phase = {}
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] in seed_colors:
                    phase[(r + c) % 2] = int(grid[r, c])
        
        # If incomplete mapping, try to complete it
        if 0 in phase and 1 not in phase:
            other = [c for c in seed_colors if c != phase[0]]
            if other: phase[1] = other[0]
        elif 1 in phase and 0 not in phase:
            other = [c for c in seed_colors if c != phase[1]]
            if other: phase[0] = other[0]
        elif not phase:
            # Fallback
            return grid
            
        out = grid.copy()
        q = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] in seed_colors:
                    q.append((r, c))
                    
        head = 0
        while head < len(q):
            r, c = q[head]
            head += 1
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if out[nr, nc] == 0:
                        out[nr, nc] = phase.get((nr + nc) % 2, 0)
                        q.append((nr, nc))
        return out

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
