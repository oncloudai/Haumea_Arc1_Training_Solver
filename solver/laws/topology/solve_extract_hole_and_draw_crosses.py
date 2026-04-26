import numpy as np
from typing import List, Optional
from collections import Counter

def solve_extract_hole_and_draw_crosses(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies the 'background' color (most frequent).
    Finds the largest rectangular region (the 'hole') where this background color 
    is entirely absent.
    The output is a grid of the same size as the hole, with background color as base.
    Identifies 'seeds' (any non-background pixel) within the hole.
    For each seed, draws a full cross of the 'seed color' (the most frequent 
    non-background, non-zero color in the grid).
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # 1. Background color
        counts = Counter(grid.flatten())
        bg_color = counts.most_common(1)[0][0]
        
        # 2. Find the largest rectangular hole (no bg_color)
        mask = (grid != bg_color).astype(int)
        # We need the largest rectangle of 1s in this mask
        max_area = -1
        best_hole = None # r1, r2, c1, c2
        
        heights = np.zeros(w, dtype=int)
        for r in range(h):
            for c in range(w):
                if mask[r, c] == 1:
                    heights[c] += 1
                else:
                    heights[c] = 0
            stack = []
            temp_heights = list(heights) + [0]
            for i, hv in enumerate(temp_heights):
                while stack and temp_heights[stack[-1]] >= hv:
                    H_rect = temp_heights[stack.pop()]
                    W_rect = i if not stack else i - stack[-1] - 1
                    area = H_rect * W_rect
                    if area > max_area:
                        max_area = area
                        best_hole = (r - H_rect + 1, r, i - W_rect, i - 1)
                stack.append(i)
                
        if not best_hole: return None
        r1, r2, c1, c2 = best_hole
        oh, ow = r2 - r1 + 1, c2 - c1 + 1
        
        # 3. Seed color
        # Most frequent non-bg, non-zero color in the grid
        other_counts = Counter(grid.flatten())
        if bg_color in other_counts: del other_counts[bg_color]
        if 0 in other_counts: del other_counts[0]
        
        if not other_counts:
            seed_color = 0 # Default if no other color
        else:
            seed_color = other_counts.most_common(1)[0][0]
            
        # 4. Identify seeds and their specific colors
        seeds = []
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if grid[r, c] != bg_color:
                    # In some tasks (like T0), the seed color is the pixel's own color.
                    # In others (like T1), it's a different color.
                    # Let's use the pixel's color if it's not 0, else the global seed_color.
                    sc = grid[r, c] if grid[r, c] != 0 else seed_color
                    seeds.append((r - r1, c - c1, sc))
                    
        # 5. Construct output
        out = np.full((oh, ow), bg_color, dtype=int)
        # Draw crosses
        # Sort seeds to ensure consistent overlapping (optional)
        for sr, sc, s_color in seeds:
            out[sr, :] = s_color
            out[:, sc] = s_color
            
        return out

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
