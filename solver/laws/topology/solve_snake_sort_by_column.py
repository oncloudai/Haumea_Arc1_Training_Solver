import numpy as np
from typing import List, Optional

def solve_snake_sort_by_column(solver) -> Optional[List[np.ndarray]]:
    consistent = True; found_any = False
    for inp, out in solver.pairs:
        if out.shape != (3, 3): consistent = False; break
        
        # Get non-zero pixels
        coords = np.argwhere(inp != 0)
        # Sort by column, then row
        sorted_indices = sorted(coords, key=lambda x: (x[1], x[0]))
        colors = [inp[r, c] for r, c in sorted_indices]
        
        pred = np.zeros((3, 3), dtype=int)
        # Snake order mapping to pred indices
        snake_indices = [
            (0, 0), (0, 1), (0, 2),
            (1, 2), (1, 1), (1, 0),
            (2, 0), (2, 1), (2, 2)
        ]
        
        for i in range(min(len(colors), 9)):
            r, c = snake_indices[i]
            pred[r, c] = colors[i]
            
        if not np.array_equal(pred, out):
            consistent = False; break
        found_any = True
        
    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            coords = np.argwhere(ti != 0)
            sorted_indices = sorted(coords, key=lambda x: (x[1], x[0]))
            colors = [ti[r, c] for r, c in sorted_indices]
            res = np.zeros((3, 3), dtype=int)
            snake_indices = [(0,0),(0,1),(0,2),(1,2),(1,1),(1,0),(2,0),(2,1),(2,2)]
            for i in range(min(len(colors), 9)):
                r, c = snake_indices[i]
                res[r, c] = colors[i]
            results.append(res)
        return results
    return None
