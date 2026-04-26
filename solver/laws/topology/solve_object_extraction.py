import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_object_extraction(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        for conn in [4, 8]:
            for rank in range(5):
                consistent = True; found_any = False
                for inp, out in solver.pairs:
                    blobs = get_blobs(inp, bg, conn)
                    blobs = sorted(blobs, key=lambda b: (-b['size'], b['top_left'][0], b['top_left'][1]))
                    if rank >= len(blobs): consistent = False; break
                    target_blob = blobs[rank]; tr, tc = target_blob['top_left']
                    th, tw = target_blob['coords'].max(axis=0) - target_blob['top_left'] + 1
                    sub = inp[tr:tr+th, tc:tc+tw]
                    if not np.array_equal(sub, out): consistent = False; break
                    found_any = True
                if consistent and found_any:
                    results = []
                    for ti in solver.test_in:
                        blobs = get_blobs(ti, bg, conn)
                        blobs = sorted(blobs, key=lambda b: (-b['size'], b['top_left'][0], b['top_left'][1]))
                        if rank < len(blobs):
                            target = blobs[rank]; tr, tc = target['top_left']
                            th, tw = target['coords'].max(axis=0) - target['top_left'] + 1
                            results.append(ti[tr:tr+th, tc:tc+tw])
                        else: break
                    if len(results) == len(solver.test_in): return results
    return None
