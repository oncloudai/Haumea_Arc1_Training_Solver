import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_color_by_max_object_size(solver) -> Optional[List[np.ndarray]]:
    def get_max_colors(blobs, max_size):
        color_min_c = {}
        for b in blobs:
            if len(b['coords']) == max_size:
                c = b['color']
                min_c = b['coords'][:, 1].min()
                if c not in color_min_c or min_c < color_min_c[c]:
                    color_min_c[c] = min_c
        return [c for c, mc in sorted(color_min_c.items(), key=lambda x: x[1])]

    for conn in [8, 4]:
        consistent = True; found_any = False
        for inp, out in solver.pairs:
            blobs = get_blobs(inp, 0, conn)
            if not blobs: consistent = False; break
            
            sizes = [len(b['coords']) for b in blobs]
            max_size = max(sizes)
            max_colors = get_max_colors(blobs, max_size)
            
            expected_out = np.zeros((max_size, len(max_colors)), dtype=int)
            for r in range(max_size):
                for c in range(len(max_colors)):
                    expected_out[r, c] = max_colors[c]
            
            if not np.array_equal(expected_out, out):
                consistent = False; break
            found_any = True
            
        if consistent and found_any:
            results = []
            for ti in solver.test_in:
                blobs = get_blobs(ti, 0, conn)
                if not blobs: results.append(ti.copy()); continue
                sizes = [len(b['coords']) for b in blobs]
                max_size = max(sizes)
                max_colors = get_max_colors(blobs, max_size)
                res = np.zeros((max_size, len(max_colors)), dtype=int)
                for r in range(max_size):
                    for c in range(len(max_colors)): res[r, c] = max_colors[c]
                results.append(res)
            return results
    return None
