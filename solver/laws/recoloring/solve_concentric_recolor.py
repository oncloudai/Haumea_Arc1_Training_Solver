import numpy as np
from typing import List, Optional

def solve_concentric_recolor(solver) -> Optional[List[np.ndarray]]:
    def get_layers(grid):
        h, w = grid.shape
        layers = []
        for d in range((min(h, w) + 1) // 2):
            coords = []
            # top
            for c in range(d, w - d): coords.append((d, c))
            # right
            for r in range(d + 1, h - d): coords.append((r, w - 1 - d))
            # bottom
            if h - 1 - d > d:
                for c in range(w - 2 - d, d - 1, -1): coords.append((h - 1 - d, c))
            # left
            if w - 1 - d > d:
                for r in range(h - 2 - d, d, -1): coords.append((r, d))
            if coords:
                seen = set(); final_coords = []
                for p in coords:
                    if p not in seen: final_coords.append(p); seen.add(p)
                layers.append(final_coords)
        return layers

    # Try various cycle lengths and offsets
    for cycle_len in range(2, 6):
        for offset in range(-(cycle_len - 1), cycle_len):
            if offset == 0: continue
            consistent = True; found_change = False
            for inp, out in solver.pairs:
                if inp.shape != out.shape: consistent = False; break
                layers = get_layers(inp)
                if not layers: consistent = False; break
                N = len(layers)
                pred = np.zeros_like(inp)
                for i in range(N):
                    src_idx = (i + offset) % cycle_len
                    if src_idx >= N:
                        # Fallback if cycle is longer than layers available
                        # But in bda2d7a6, N >= cycle_len (3)
                        consistent = False; break
                    c = inp[layers[src_idx][0][0], layers[src_idx][0][1]]
                    for r, c_idx in layers[i]: pred[r, c_idx] = c
                if not np.array_equal(pred, out): consistent = False; break
                if not np.array_equal(pred, inp): found_change = True
            
            if consistent and found_change:
                results = []
                for ti in solver.test_in:
                    layers = get_layers(ti); N = len(layers); res = np.zeros_like(ti)
                    if not layers: results.append(ti); continue
                    for i in range(N):
                        src_idx = (i + offset) % cycle_len
                        # If src_idx >= N, we might need a better fallback
                        # For now, just use modulo N as fallback
                        s_idx = src_idx if src_idx < N else src_idx % N
                        c = ti[layers[s_idx][0][0], layers[s_idx][0][1]]
                        for r, c_idx in layers[i]: res[r, c_idx] = c
                    results.append(res)
                return results
    return None
