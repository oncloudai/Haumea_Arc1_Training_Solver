
import numpy as np
from typing import List, Optional

def solve_vertical_rectangle_separation(solver) -> Optional[List[np.ndarray]]:
    """
    Find colored rectangles. Sort them by horizontal position.
    Shift them vertically so they are non-overlapping with a gap of 1.
    The order seems to be: left-most stays at some height, others shift up.
    """
    def get_rects(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        labeled = np.zeros_like(grid)
        curr = 1
        rects = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0 and labeled[r, c] == 0:
                    coords = []
                    q = [(r, c)]; labeled[r, c] = curr
                    color = grid[r, c]
                    while q:
                        cr, cc = q.pop(0)
                        coords.append((cr, cc))
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and labeled[nr, nc] == 0 and grid[nr, nc] == color:
                                labeled[nr, nc] = curr; q.append((nr, nc))
                    coords = np.array(coords)
                    r_min, c_min = coords.min(axis=0)
                    r_max, c_max = coords.max(axis=0)
                    rects.append({
                        'color': color,
                        'r_min': r_min, 'r_max': r_max,
                        'c_min': c_min, 'c_max': c_max,
                        'h': r_max - r_min + 1,
                        'w': c_max - c_min + 1,
                        'coords': coords
                    })
                    curr += 1
        return rects

    def apply_logic(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        rects = get_rects(grid)
        if not rects: return grid, False
        
        # Sort by horizontal position
        rects.sort(key=lambda x: x['c_min'])
        
        out = np.zeros_like(grid)
        
        # In Ex 0: 
        # Left (1): r 11-14 -> output r 7-10 (shifted up 4)
        # Middle (2): r 13-14 -> output r 11-12 (shifted up 2?)
        # Right (4): r 11-14 -> output r 7-10 (shifted up 4)
        
        # Wait, the rule might be simpler: all rectangles are pulled to the same 
        # base level or separated?
        # Let's re-examine Ex 0 output.
        # Color 1 is at r 7-10.
        # Color 2 is at r 11-12.
        # Color 4 is at r 7-10.
        # Color 2 was AT THE BOTTOM (r 13-14).
        
        # New hypothesis: The rectangles are "pushed" to the bottom, but 
        # they cannot overlap. If they overlap horizontally, they must stack.
        # No, Color 1 and Color 4 do NOT overlap horizontally, but Color 2 
        # overlaps BOTH.
        
        # Let's try stacking: 
        # Sort by horizontal overlaps?
        # This looks like they are moved to a specific middle row and then 
        # stacked if they overlap.
        
        # Let's try: move everything to the bottom-most available rows 
        # that don't cause overlap.
        
        rects.sort(key=lambda x: x['r_min']) # Sort by vertical to preserve some order
        
        occupied = np.zeros((rows, cols), dtype=bool)
        
        # We need to find the "base" row. 
        # In Ex 0, the bottom-most row of output is 12 (for color 2).
        # In Ex 0, rows 13 and 14 are empty in output.
        
        # Let's try to place them from left to right, but check vertical stacking.
        # Actually, let's try to match the output's specific rows.
        # Color 1 and 4 are at 7-10. Color 2 is at 11-12.
        # Total height used is 7 to 12. 
        # This is exactly the height of (Color 1 or 4) + (Color 2) = 4 + 2 = 6.
        
        # Yes! Stacking. If they share columns, they stack.
        # Color 1 (c 1-2) and Color 2 (c 4-7) and Color 4 (c 9-12).
        # Wait, in Ex 0:
        # 1: c 1-2
        # 2: c 4-7
        # 4: c 9-12
        # They DON'T overlap horizontally! 
        # 1-2, 4-7, 9-12 are separate.
        # So why is 2 lower than 1 and 4?
        
        # Maybe they are aligned to the bottom, but 1 and 4 are shifted up?
        # Or 2 is shifted down?
        
        # Let's check Ex 1.
        # Input: 
        # 1 (c 1-2, r 10-13)
        # 4 (c 9-12, r 10-13)
        # 3 (c 5-7, r 12-13)
        # Output:
        # 1 & 4 at r 6-9
        # 3 at r 10-11
        
        # Rule: 1 and 4 are always at the same height. 3 (or 2) is below them.
        # 1 and 4 are the outer ones. 2/3 is the middle one.
        # The middle one stays lower?
        
        return grid, False

    return None
