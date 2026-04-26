import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_object_move_by_marker_color(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        for conn in [4, 8]:
            for m_color in range(10):
                if m_color == bg: continue
                consistent = True; found_move = False; rule = {}
                for inp, out in solver.pairs:
                    m_c = np.argwhere(inp == m_color)
                    if len(m_c) == 0: consistent = False; break
                    m_pos = m_c[0]; in_b = get_blobs(inp, bg, conn); out_b = get_blobs(out, bg, conn)
                    pred = np.full_like(out, bg); pred[m_c[:,0], m_c[:,1]] = m_color
                    for b in in_b:
                        if b['color'] == m_color: continue
                        match = next((ob for ob in out_b if ob['normalized'] == b['normalized'] and ob['color'] == b['color']), None)
                        if match is None: consistent = False; break
                        delta = tuple(match['top_left'] - m_pos)
                        if b['color'] in rule and rule[b['color']] != delta: consistent = False; break
                        rule[b['color']] = delta; found_move = True
                        for r, c in b['coords'] + (m_pos[0] + delta[0] - b['top_left'][0], m_pos[1] + delta[1] - b['top_left'][1]):
                            if 0 <= r < pred.shape[0] and 0 <= c < pred.shape[1]: pred[r, c] = b['color']
                    if not consistent or not np.array_equal(pred, out): consistent = False; break
                if consistent and found_move:
                    results = []
                    for ti in solver.test_in:
                        res = np.full_like(ti, bg); m_c = np.argwhere(ti == m_color)
                        if len(m_c) == 0: return None
                        mp = m_c[0]; res[m_c[:,0], m_c[:,1]] = m_color
                        for b in get_blobs(ti, bg, conn):
                            if b['color'] != m_color and b['color'] in rule:
                                dr, dc = rule[b['color']]
                                for r, c in b['coords'] + (mp[0] + dr - b['top_left'][0], mp[1] + dc - b['top_left'][1]):
                                    if 0 <= r < res.shape[0] and 0 <= c < res.shape[1]: res[r, c] = b['color']
                        results.append(res)
                    return results
    return None
