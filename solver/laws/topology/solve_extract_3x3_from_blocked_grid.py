
import numpy as np
from typing import List, Optional

def solve_extract_3x3_from_blocked_grid(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a 3x3 arrangement of blocks in the input.
    The output is a 3x3 grid where each cell represents a block.
    The color of the cell in the output is determined by the presence 
    of a 'noise' color in or around the corresponding block.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        
        # Find all non-zero colors
        unique, counts = np.unique(grid, return_counts=True)
        colors = {u: c for u, c in zip(unique, counts) if u != 0}
        if len(colors) < 2: return None
        
        # Sort by frequency: most frequent is likely the 'block' color
        sorted_colors = sorted(colors.items(), key=lambda x: x[1], reverse=True)
        block_color = sorted_colors[0][0]
        # The 'noise' color is the one we want to map to the output
        noise_color = sorted_colors[1][0]
        
        # Find all coordinates of the block color
        br, bc = np.where(grid == block_color)
        if len(br) == 0: return None
        
        min_r, max_r = br.min(), br.max()
        min_c, max_c = bc.min(), bc.max()
        
        # Divide the bounding box into a 3x3 grid
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        
        # In 5ad4f10b, blocks might not be perfectly divided by 3
        # Let's try to find the 3x3 boundaries by gaps or equal size
        
        out = np.zeros((3, 3), dtype=int)
        
        # Try equal division first
        for i in range(3):
            for j in range(3):
                r_start = min_r + (i * h) // 3
                r_end = min_r + ((i + 1) * h) // 3
                c_start = min_c + (j * w) // 3
                c_end = min_c + ((j + 1) * w) // 3
                
                # Check for noise color in or near this block
                # Looking slightly outside the block too (e.g., padding of 2)
                r_low = max(0, r_start - 2)
                r_high = min(rows, r_end + 2)
                c_low = max(0, c_start - 2)
                c_high = min(cols, c_end + 2)
                
                sub = grid[r_low:r_high, c_low:c_high]
                if np.any(sub == noise_color):
                    out[i, j] = noise_color
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
