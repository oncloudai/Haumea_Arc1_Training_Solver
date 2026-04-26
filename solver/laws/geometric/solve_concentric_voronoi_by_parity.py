import numpy as np
from typing import List, Optional

def solve_concentric_voronoi_by_parity(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies all non-zero pixels as seeds.
    For each grid pixel, finds the closest seed using Chebyshev distance.
    Breaks ties with Manhattan distance.
    Colors the pixel with the winner's color IF the distance is even.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        seeds = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0:
                    seeds.append(((r, c), int(grid[r, c])))
        
        if not seeds: return grid
        
        output = np.zeros((rows, cols), dtype=int)
        for r in range(rows):
            for c in range(cols):
                min_cheb = float('inf')
                tied_seeds = []
                
                for (sr, sc), color in seeds:
                    d = max(abs(r - sr), abs(c - sc))
                    if d < min_cheb:
                        min_cheb = d
                        tied_seeds = [((sr, sc), color)]
                    elif d == min_cheb:
                        tied_seeds.append(((sr, sc), color))
                
                winner = None
                if len(tied_seeds) == 1:
                    winner = tied_seeds[0]
                else:
                    # Tie-breaking with Manhattan distance
                    min_manh = float('inf')
                    tied_manh = []
                    for (sr, sc), color in tied_seeds:
                        m = abs(r - sr) + abs(c - sc)
                        if m < min_manh:
                            min_manh = m
                            tied_manh = [((sr, sc), color)]
                        elif m == min_manh:
                            tied_manh.append(((sr, sc), color))
                    
                    if len(tied_manh) == 1:
                        winner = tied_manh[0]
                
                if winner and min_cheb % 2 == 0:
                    output[r, c] = winner[1]
        return output

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
