import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_marker_pattern_stamping(solver) -> Optional[List[np.ndarray]]:
    bg = 0
    marker_colors = {1, 2, 3, 4, 6}
    consistent = True; rules = {}
    
    for inp, out in solver.pairs:
        labeled, n = label(np.isin(inp, list(marker_colors)), structure=np.ones((3,3)))
        for i in range(1, n+1):
            m_mask = (labeled == i); m_coords = np.argwhere(m_mask)
            m_found = sorted([(int(inp[r,c]), r, c) for r,c in m_coords])
            m_ref_r, m_ref_c = m_found[0][1], m_found[0][2]
            m_rel = tuple(sorted([(c, r - m_ref_r, col - m_ref_c) for c, r, col in m_found]))
            
            # Find payload at this marker set in OUTPUT
            labeled_out, n_out = label(out == 8, structure=np.ones((3,3)))
            p_rel = []
            for k in range(1, n_out+1):
                p_mask = (labeled_out == k); p_coords = np.argwhere(p_mask)
                if np.any(np.abs(p_coords[:, None, :] - m_coords[None, :, :]).sum(axis=2) <= 1):
                    p_rel = tuple(sorted([(8, r - m_ref_r, col - m_ref_c) for r, col in p_coords]))
                    break
            
            if p_rel:
                if m_rel in rules and rules[m_rel] != p_rel: consistent = False; break
                rules[m_rel] = p_rel
        if not consistent: break
        
    if consistent and rules:
        def process(grid):
            res = grid.copy(); h, w = grid.shape
            res[grid == 8] = bg
            labeled, n = label(np.isin(grid, list(marker_colors)), structure=np.ones((3,3)))
            for i in range(1, n+1):
                m_mask = (labeled == i); m_coords = np.argwhere(m_mask)
                m_found = sorted([(int(grid[r,c]), r, c) for r,c in m_coords])
                m_ref_r, m_ref_c = m_found[0][1], m_found[0][2]
                m_rel = tuple(sorted([(c, r - m_ref_r, col - m_ref_c) for c, r, col in m_found]))
                if m_rel in rules:
                    for pc, dr, dc in rules[m_rel]:
                        nr, nc = m_ref_r + dr, m_ref_c + dc
                        if 0 <= nr < h and 0 <= nc < w: res[nr, nc] = pc
            return res
        for inp, out in solver.pairs:
            if not np.array_equal(process(inp), out): consistent = False; break
        if consistent:
            return [process(ti) for ti in solver.test_in]
    return None
