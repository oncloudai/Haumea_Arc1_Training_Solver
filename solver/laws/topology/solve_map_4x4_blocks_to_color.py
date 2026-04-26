import numpy as np
from typing import List, Optional

def solve_map_4x4_blocks_to_color(solver) -> Optional[List[np.ndarray]]:
    """
    Learns a mapping from 4x4 subgrid patterns (consisting of color 5 and 0)
    to a result color. The input is 4x14, containing three 4x4 blocks.
    The output is 3x3, where each row is the color mapped from the corresponding block.
    """
    def extract_blocks(grid):
        grid = np.array(grid)
        if grid.shape != (4, 14): return None
        # Blocks at C0-3, C5-8, C10-13
        b1 = grid[:, 0:4]
        b2 = grid[:, 5:9]
        b3 = grid[:, 10:14]
        return [b1, b2, b3]

    mapping = {}
    for inp, out in solver.pairs:
        blocks = extract_blocks(inp)
        if blocks is None: return None
        out = np.array(out)
        if out.shape != (3, 3): return None
        
        # Each row in output is a color
        for i in range(3):
            block_bytes = blocks[i].tobytes()
            color = out[i, 0] # Assuming entire row is same color
            if block_bytes in mapping and mapping[block_bytes] != color:
                return None
            mapping[block_bytes] = color
            
    def apply_logic(grid):
        blocks = extract_blocks(grid)
        if blocks is None: return None
        
        out = np.zeros((3, 3), dtype=int)
        for i in range(3):
            block_bytes = blocks[i].tobytes()
            if block_bytes not in mapping:
                return None
            out[i, :] = mapping[block_bytes]
        return out

    # Check if logic works for all pairs
    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
