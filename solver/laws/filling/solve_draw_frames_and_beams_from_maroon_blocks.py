import numpy as np
from typing import List, Optional

def solve_draw_frames_and_beams_from_maroon_blocks(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies connected components of color 9 (maroon).
    For each block:
    1. Calculates a 'thickness' based on its size (max(h, w) // 2).
    2. Draws a 'frame' of color 3 around the block with that thickness.
    3. Shoots a 'beam' of color 1 downwards from the bottom of the frame.
    Layers the components: Beams (bottom), then Frames, then Maroon Blocks (top).
    """
    def get_blocks(grid):
        rows, cols = grid.shape
        labeled = np.zeros_like(grid)
        curr = 0
        blocks = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 9 and labeled[r, c] == 0:
                    curr += 1
                    q = [(r, c)]
                    labeled[r, c] = curr
                    pixels = []
                    while q:
                        cr, cc = q.pop(0)
                        pixels.append((cr, cc))
                        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                            nr, nc = cr+dr, cc+dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if grid[nr, nc] == 9 and labeled[nr, nc] == 0:
                                    labeled[nr, nc] = curr
                                    q.append((nr, nc))
                    coords = np.array(pixels)
                    blocks.append({
                        'pixels': pixels,
                        'bbox': (coords[:,0].min(), coords[:,0].max(), coords[:,1].min(), coords[:,1].max())
                    })
        return blocks

    def apply_logic(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        blocks = get_blocks(grid)
        if not blocks: return grid
        
        output = np.zeros_like(grid)
        
        # 1. Beams
        for b in blocks:
            r1, r2, c1, c2 = b['bbox']
            thickness = max(r2 - r1 + 1, c2 - c1 + 1) // 2
            beam_start = r2 + thickness + 1
            if beam_start < rows:
                output[beam_start:, c1:c2+1] = 1
                
        # 2. Frames
        for b in blocks:
            r1, r2, c1, c2 = b['bbox']
            thickness = max(r2 - r1 + 1, c2 - c1 + 1) // 2
            f_r1, f_r2 = max(0, r1 - thickness), min(rows - 1, r2 + thickness)
            f_c1, f_c2 = max(0, c1 - thickness), min(cols - 1, c2 + thickness)
            output[f_r1:f_r2+1, f_c1:f_c2+1] = 3
            
        # 3. Maroon blocks
        for b in blocks:
            for r, c in b['pixels']:
                output[r, c] = 9
                
        return output

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
