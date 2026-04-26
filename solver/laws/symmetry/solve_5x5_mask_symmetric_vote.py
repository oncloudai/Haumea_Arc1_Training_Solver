import numpy as np
from typing import List, Optional

def get_symmetric_points_ff805c23(r, c, N, M):
    # Standard 8-fold symmetry for a square grid
    # N x N assumed for rotational/reflective symmetry
    # 4-fold rotations:
    p1 = (r, c)
    p2 = (c, N - 1 - r)
    p3 = (N - 1 - r, N - 1 - c)
    p4 = (N - 1 - c, r)
    
    # Reflections
    p5 = (r, N - 1 - c)
    p6 = (N - 1 - r, c)
    p7 = (c, r)
    p8 = (N - 1 - c, N - 1 - r)
    
    return [p1, p2, p3, p4, p5, p6, p7, p8]

def solve_grid_ff805c23(input_grid):
    grid = np.array(input_grid)
    N, M = grid.shape
    
    # 1. Find the 5x5 mask (color 1)
    mask_coords = np.argwhere(grid == 1)
    if len(mask_coords) == 0:
        return np.zeros((5, 5), dtype=int)
        
    r_min, r_max = mask_coords[:, 0].min(), mask_coords[:, 0].max()
    c_min, c_max = mask_coords[:, 1].min(), mask_coords[:, 1].max()
    
    # Output is 5x5
    h, w = 5, 5
    output = np.zeros((h, w), dtype=int)
    
    for dr in range(h):
        for dc in range(w):
            r, c = r_min + dr, c_min + dc
            
            # Find all symmetric points to (r, c)
            sym_points = get_symmetric_points_ff805c23(r, c, N, M)
            
            # Vote for the color
            votes = {}
            for pr, pc in sym_points:
                if 0 <= pr < N and 0 <= pc < M:
                    color = grid[pr, pc]
                    if color != 1: # Don't count the mask itself
                        votes[color] = votes.get(color, 0) + 1
            
            if votes:
                best_color = max(votes, key=votes.get)
                output[dr, dc] = best_color
                
    return output

def solve_5x5_mask_symmetric_vote(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_ff805c23(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_ff805c23(ti) for ti in solver.test_in]
    return None
