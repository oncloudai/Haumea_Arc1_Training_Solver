
import numpy as np
from typing import List, Optional

def solve_move_block_with_color_as_position(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a 3x3 grid of 3x3 blocks.
    Finds the 3x3 block containing a marker color (default 4).
    The relative position (dr, dc) of the marker within its source block
    determines the destination block index (R, C) in the 3x3 grid.
    Moves the source block to the destination block and clears all other blocks.
    """
    def run_single(input_grid, marker_color):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        # Assuming fixed layout for this task (11x11, dividers at 3, 7)
        if rows != 11 or cols != 11: return None
        
        tls = [(r, c) for r in [0, 4, 8] for c in [0, 4, 8]]
        
        src_block_idx = -1
        rel_pos = None
        for i, (r, c) in enumerate(tls):
            block = grid[r:r+3, c:c+3]
            coords = np.argwhere(block == marker_color)
            if len(coords) > 0:
                src_block_idx = i
                rel_pos = tuple(coords[0])
                break
                
        if src_block_idx == -1: return None
        
        # Target block index in 3x3 grid is rel_pos
        tR, tC = rel_pos
        tr, tc = tls[tR * 3 + tC]
        sr, sc = tls[src_block_idx]
        
        output = grid.copy()
        # Clear all blocks
        for r, c in tls:
            output[r:r+3, c:c+3] = 0
            
        # Copy identity
        output[tr:tr+3, tc:tc+3] = grid[sr:sr+3, sc:sc+3]
        
        return output

    # Brute force marker color
    for color in range(1, 10):
        consistent = True
        found_any = False
        for inp, out_expected in solver.pairs:
            pred = run_single(inp, color)
            if pred is None or not np.array_equal(pred, out_expected):
                consistent = False; break
            found_any = True
        if consistent and found_any:
            return [run_single(ti, color) if run_single(ti, color) is not None else np.array(ti) for ti in solver.test_in]
            
    return None
