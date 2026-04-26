import numpy as np
from typing import List, Optional
from scipy.ndimage import label
from solver.utils import get_blobs

def solve_object_move_to_marker(solver) -> Optional[List[np.ndarray]]:
    bg = 0
    # For tasks like 0e206a2e
    for conn in [4, 8]:
        consistent = True; found_any = False; template = None
        for inp, out in solver.pairs:
            blobs = get_blobs(inp, bg, conn)
            if len(blobs) < 2: consistent = False; break
            
            # The payload is the largest blob
            blobs = sorted(blobs, key=lambda b: len(b['coords']), reverse=True)
            tmpl = blobs[0]
            markers = blobs[1:]
            
            # Find closest marker blob
            m_dists = [np.min(np.sum(np.abs(m['coords'][:, None] - tmpl['coords'][None, :]), axis=2)) for m in markers]
            target_m = markers[np.argmin(m_dists)]; m_tl = target_m['coords'].min(axis=0)
            m_pattern = tuple(sorted([(int(inp[r,c]), r-m_tl[0], c-m_tl[1]) for r,c in target_m['coords']]))
            p_rel = tuple(sorted([(r-m_tl[0], c-m_tl[1]) for r,c in tmpl['coords']]))
            
            if template is None: template = (m_pattern, p_rel)
            elif template != (m_pattern, p_rel): consistent = False; break
            
            # Identify all marker groups in the grid
            labeled_m, n_m = label(np.isin(inp, [int(inp[r,c]) for r,c in np.argwhere(inp != bg) if (r,c) not in [tuple(x) for x in tmpl['coords']]]), structure=np.ones((3,3)))
            pred = inp.copy()
            for i in range(1, n_m + 1):
                m_coords = np.argwhere(labeled_m == i); tl = m_coords.min(axis=0)
                pat = tuple(sorted([(int(inp[r,c]), r-tl[0], c-tl[1]) for r,c in m_coords]))
                if pat == m_pattern:
                    for dr, dc in p_rel:
                        nr, nc = tl[0] + dr, tl[1] + dc
                        if 0 <= nr < pred.shape[0] and 0 <= nc < pred.shape[1]:
                            pred[nr, nc] = tmpl['color']; found_any = True
            if not np.array_equal(pred, out): consistent = False; break
            
        if consistent and found_any:
            m_pattern, p_rel = template
            results = []
            for ti in solver.test_in:
                res = ti.copy(); blobs = get_blobs(ti, bg, conn); tmpl = sorted(blobs, key=lambda b: len(b['coords']), reverse=True)[0]
                m_colors = [int(ti[r,c]) for r,c in np.argwhere(ti != bg) if (r,c) not in [tuple(x) for x in tmpl['coords']]]
                labeled_m, n_m = label(np.isin(ti, m_colors), structure=np.ones((3,3)))
                for i in range(1, n_m + 1):
                    m_coords = np.argwhere(labeled_m == i); tl = m_coords.min(axis=0)
                    pat = tuple(sorted([(int(ti[r,c]), r-tl[0], c-tl[1]) for r,c in m_coords]))
                    if pat == m_pattern:
                        for dr, dc in p_rel:
                            nr, nc = tl[0] + dr, tl[1] + dc
                            if 0 <= nr < res.shape[0] and 0 <= nc < res.shape[1]: res[nr, nc] = tmpl['color']
                results.append(res)
            return results
    return None
