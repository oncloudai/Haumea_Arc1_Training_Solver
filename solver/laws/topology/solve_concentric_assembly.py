import numpy as np
from typing import List, Optional
from collections import Counter

def get_symmetries(r, c, s):
    mid = (s - 1) / 2
    dr, dc = r - mid, c - mid
    syms = [
        (dr, dc), (dr, -dc), (-dr, dc), (-dr, -dc),
        (dc, dr), (dc, -dr), (-dc, dr), (-dc, -dr)
    ]
    return sorted(list(set([(int(round(x + mid)), int(round(y + mid))) for x, y in syms])))

def get_all_valid_ds(pixels):
    """Returns list of (d, mask) for ALL valid d values (not just the first)."""
    if len(pixels) == 0:
        return []
    pixels = np.array(pixels)
    results = []

    for d in range(16):
        if d == 0:
            if len(np.unique(pixels, axis=0)) == 1:
                results.append((d, {(0, 0)}))
            continue

        candidates = None
        for r, c in pixels:
            current_p = set()
            for i in range(-d, d + 1):
                current_p.add((r - d, c + i))
                current_p.add((r + d, c + i))
                current_p.add((r + i, c - d))
                current_p.add((r + i, c + d))
            if candidates is None:
                candidates = current_p
            else:
                candidates &= current_p
            if not candidates:
                break

        if candidates:
            # We pick the first candidate as the center r0, c0
            r0, c0 = next(iter(candidates))
            mask = set()
            s = 2 * d + 1
            all_on_perimeter = True
            for r, c in pixels:
                dr, dc = abs(r - r0), abs(c - c0)
                if max(dr, dc) != d:
                    all_on_perimeter = False
                    break
                rel_r, rel_c = r - (r0 - d), c - (c0 - d)
                for mr, mc in get_symmetries(rel_r, rel_c, s):
                    mask.add((mr, mc))
            if all_on_perimeter:
                results.append((d, mask))

    return results

def bbox_area(pixels):
    pixels = np.array(pixels)
    if len(pixels) == 0: return 0
    r_min, c_min = pixels.min(axis=0)
    r_max, c_max = pixels.max(axis=0)
    return (r_max - r_min + 1) * (c_max - c_min + 1)

def solve_concentric_assembly(solver) -> Optional[List[np.ndarray]]:
    def run_single(grid):
        grid = np.array(grid)
        unique_colors = np.unique(grid)
        counts = {c: np.sum(grid == c) for c in unique_colors}
        bg = max(counts, key=counts.get)

        # Collect all valid (d, mask) options per color, plus bounding box area
        layer_options = []
        for c in unique_colors:
            if c == bg:
                continue
            pixels = np.argwhere(grid == c)
            options = get_all_valid_ds(pixels)
            if options:
                area = bbox_area(pixels)
                layer_options.append((c, options, area))

        if not layer_options:
            return None

        # Sort: fewer options first (must-assign shapes get priority),
        # then by min valid d, then by bbox area (smaller area -> smaller d / inner ring)
        layer_options.sort(key=lambda x: (len(x[1]), x[1][0][0], x[2]))

        # Greedy assignment: each color takes its smallest available d
        layers = []
        used_ds = set()
        for c, options, area in layer_options:
            assigned = False
            for d, mask in sorted(options, key=lambda x: x[0]):
                if d not in used_ds:
                    layers.append((c, d, mask))
                    used_ds.add(d)
                    assigned = True
                    break
            if not assigned:
                # If we couldn't assign any d, this logic might fail for this grid
                return None

        if not layers:
            return None

        max_d = max(l[1] for l in layers)
        N = 2 * max_d + 1
        out = np.full((N, N), bg)

        for c, d, mask in layers:
            offset = max_d - d
            for r, ci in mask:
                if 0 <= r < (2 * d + 1) and 0 <= ci < (2 * d + 1):
                    # Only fill the perimeter of the d-ring (or center if d==0)
                    if d == 0 or r == 0 or r == 2 * d or ci == 0 or ci == 2 * d:
                        out[offset + r, offset + ci] = c
        return out

    results = []
    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or pred.shape != out_expected.shape or not np.array_equal(pred, out_expected):
            return None
    
    for ti in solver.test_in:
        results.append(run_single(ti))
    return results
