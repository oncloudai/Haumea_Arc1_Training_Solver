import numpy as np
from collections import Counter
from typing import List, Optional

def solve_grid_72322fa7(grid):
    grid = np.array(grid)
    h, w = grid.shape
    colors = np.unique(grid)
    colors = colors[colors > 0]
    
    # Discovery phase: Find consistent local patterns per (Seed, Surround) pair
    # (sc, shc) -> offsets
    patterns = {}
    
    for sc in colors:
        for shc in colors:
            if sc == shc: continue
            
            seeds = np.argwhere(grid == sc)
            candidate_offset_sets = []
            for r, c in seeds:
                offsets = []
                # Check 5x5 area
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == shc:
                            offsets.append((dr, dc))
                if offsets:
                    candidate_offset_sets.append(frozenset(offsets))
            
            if not candidate_offset_sets: continue
            
            # For each pair (sc, shc), find the most frequent "maximal" offset set.
            # In ARC, the most complete instance tells us the rule.
            # We look for a set that appears as a subset in most instances.
            
            # Actually, just take the set that appears most frequently among the seeds.
            counts = Counter(candidate_offset_sets)
            # Pick the one that is most common, or if tied, the largest.
            best_offsets, _ = max(counts.items(), key=lambda x: (x[1], len(x[0])))
            
            # We only accept it if it's a 1-to-many relationship (mostly)
            # or if it's symmetric.
            patterns[(int(sc), int(shc))] = best_offsets

    # Filter patterns: ARC usually has directed stamps.
    # If sc is seed for shc, then shc is probably not seed for sc.
    # Resolve by choosing the one with MORE offsets.
    final_patterns = {}
    processed = set()
    for (a, b), offsets in patterns.items():
        if (a, b) in processed: continue
        inv = (b, a)
        if inv in patterns:
            if len(offsets) >= len(patterns[inv]):
                final_patterns[(a, b)] = offsets
            else:
                final_patterns[inv] = patterns[inv]
            processed.add(inv)
        else:
            final_patterns[(a, b)] = offsets
        processed.add((a, b))

    # Application phase
    output_grid = grid.copy()
    for _ in range(5):
        changed = False
        for (sc, shc), offsets in final_patterns.items():
            # 1. Fill surroundings from seeds
            seeds = np.argwhere(output_grid == sc)
            for r, c in seeds:
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if output_grid[nr, nc] == 0:
                            output_grid[nr, nc] = shc
                            changed = True
            
            # 2. Fill seeds from surroundings
            for r in range(h):
                for c in range(w):
                    if output_grid[r, c] != 0 and output_grid[r, c] != sc: continue
                    # Check if all surrounding pixels are there
                    match = True
                    for dr, dc in offsets:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if output_grid[nr, nc] != shc:
                                match = False; break
                        else: match = False; break
                    if match and output_grid[r, c] == 0:
                        output_grid[r, c] = sc
                        changed = True
        if not changed: break
        
    return output_grid

def solve_seed_pattern_inference(solver) -> Optional[List[np.ndarray]]:
    # Task 72322fa7
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_72322fa7(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_72322fa7(ti) for ti in solver.test_in]
    return None
