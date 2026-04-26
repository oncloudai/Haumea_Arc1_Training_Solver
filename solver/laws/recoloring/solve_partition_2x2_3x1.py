import numpy as np
from typing import List, Optional

def solve_partition_2x2_3x1(solver) -> Optional[List[np.ndarray]]:
    def find_partition(pixels, h, w):
        if not pixels: return {}
        
        # Pick the first pixel in deterministic order
        r, c = sorted(list(pixels))[0]
        
        # Try all 2x2 blocks that could cover (r, c)
        for dr0 in range(2):
            for dc0 in range(2):
                # Potential 2x2 top-left
                tlr, tlc = r - dr0, c - dc0
                block = []
                for dr in range(2):
                    for dc in range(2):
                        block.append((tlr + dr, tlc + dc))
                if all(p in pixels for p in block):
                    new_pixels = pixels.copy()
                    for p in block: new_pixels.remove(p)
                    res = find_partition(new_pixels, h, w)
                    if res is not None:
                        for p in block: res[p] = 8
                        return res

        # Try all 1x3 blocks that could cover (r, c)
        for dc0 in range(3):
            tlr, tlc = r, c - dc0
            block = []
            for dc in range(3):
                block.append((tlr, tlc + dc))
            if all(p in pixels for p in block):
                new_pixels = pixels.copy()
                for p in block: new_pixels.remove(p)
                res = find_partition(new_pixels, h, w)
                if res is not None:
                    for p in block: res[p] = 2
                    return res

        # Try all 3x1 blocks that could cover (r, c)
        for dr0 in range(3):
            tlr, tlc = r - dr0, c
            block = []
            for dr in range(3):
                block.append((tlr + dr, tlc))
            if all(p in pixels for p in block):
                new_pixels = pixels.copy()
                for p in block: new_pixels.remove(p)
                res = find_partition(new_pixels, h, w)
                if res is not None:
                    for p in block: res[p] = 2
                    return res
        return None

    consistent = True
    for inp, out in solver.pairs:
        h, w = inp.shape
        pixels = set(tuple(p) for p in np.argwhere(inp == 5))
        mapping = find_partition(pixels, h, w)
        if mapping is None:
            consistent = False; break
        pred = inp.copy()
        for (r, c), color in mapping.items():
            pred[r, c] = color
        if not np.array_equal(pred, out):
            consistent = False; break
            
    if consistent:
        results = []
        for ti in solver.test_in:
            h, w = ti.shape
            pixels = set(tuple(p) for p in np.argwhere(ti == 5))
            mapping = find_partition(pixels, h, w)
            if mapping is None: return None
            res = ti.copy()
            for (r, c), color in mapping.items():
                res[r, c] = color
            results.append(res)
        return results
    return None
