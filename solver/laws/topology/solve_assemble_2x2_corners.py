import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_assemble_2x2_corners(solver) -> Optional[List[np.ndarray]]:
    consistent = True; found_any = False
    for inp, out in solver.pairs:
        if out.shape != (4, 4): consistent = False; break
        
        blobs = get_blobs(inp, 0, 8)
        if len(blobs) != 4: consistent = False; break
        
        # Each blob should be a 2x2 corner
        corner_map = {} # 'TL', 'TR', 'BL', 'BR' -> blob
        for b in blobs:
            r_min, c_min = b['coords'].min(axis=0)
            r_max, c_max = b['coords'].max(axis=0)
            if (r_max - r_min != 1) or (c_max - c_min != 1):
                consistent = False; break
            
            # Identify which pixel is missing
            present = set([tuple(x) for x in b['coords']])
            missing = None
            for dr in [0, 1]:
                for dc in [0, 1]:
                    if (r_min + dr, c_min + dc) not in present:
                        missing = (dr, dc); break
            
            if missing == (1, 1): corner_map['TL'] = b
            elif missing == (1, 0): corner_map['TR'] = b
            elif missing == (0, 1): corner_map['BL'] = b
            elif missing == (0, 0): corner_map['BR'] = b
            else: consistent = False; break
            
        if not consistent or len(corner_map) != 4:
            consistent = False; break
            
        # Assemble 4x4
        pred = np.zeros((4, 4), dtype=int)
        for corner, b in corner_map.items():
            r_off = 0 if 'T' in corner else 2
            c_off = 0 if 'L' in corner else 2
            r_min, c_min = b['coords'].min(axis=0)
            for r, c in b['coords']:
                pred[r_off + (r - r_min), c_off + (c - c_min)] = b['color']
                
        if not np.array_equal(pred, out):
            consistent = False; break
        found_any = True
        
    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            blobs = get_blobs(ti, 0, 8)
            if len(blobs) != 4: results.append(ti.copy()); continue
            corner_map = {}
            for b in blobs:
                r_min, c_min = b['coords'].min(axis=0)
                r_max, c_max = b['coords'].max(axis=0)
                if (r_max - r_min != 1) or (c_max - c_min != 1): continue
                present = set([tuple(x) for x in b['coords']])
                missing = None
                for dr in [0, 1]:
                    for dc in [0, 1]:
                        if (r_min + dr, c_min + dc) not in present:
                            missing = (dr, dc); break
                if missing == (1, 1): corner_map['TL'] = b
                elif missing == (1, 0): corner_map['TR'] = b
                elif missing == (0, 1): corner_map['BL'] = b
                elif missing == (0, 0): corner_map['BR'] = b
            
            if len(corner_map) != 4: results.append(ti.copy()); continue
            res = np.zeros((4, 4), dtype=int)
            for corner, b in corner_map.items():
                r_off = 0 if 'T' in corner else 2
                c_off = 0 if 'L' in corner else 2
                r_min, c_min = b['coords'].min(axis=0)
                for r, c in b['coords']:
                    res[r_off + (r - r_min), c_off + (c - c_min)] = b['color']
            results.append(res)
        return results
    return None
