
import numpy as np
from typing import List, Optional

def solve_template_stamping_to_containers(solver) -> Optional[List[np.ndarray]]:
    def get_regions(grid, bg):
        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        regions = []
        for r in range(h):
            for c in range(w):
                if not visited[r, c] and grid[r, c] != bg:
                    coords = []
                    stack = [(r, c)]
                    visited[r, c] = True
                    while stack:
                        curr_r, curr_c = stack.pop()
                        coords.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] != bg:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                    regions.append(np.array(coords))
        return regions

    def process(inp):
        h, w = inp.shape
        border = np.concatenate([inp[0, :], inp[-1, :], inp[:, 0], inp[:, -1]])
        bg = int(np.bincount(border).argmax())
        
        regions_coords = get_regions(inp, bg)
        if len(regions_coords) < 2: return None
        
        regions = []
        for coords in regions_coords:
            min_r, min_c = coords.min(axis=0)
            max_r, max_c = coords.max(axis=0)
            reg_colors = np.unique([inp[r, c] for r, c in coords])
            regions.append({
                'coords': coords,
                'size': len(coords),
                'colors': reg_colors,
                'bbox': (min_r, min_c, max_r, max_c),
                'shape': (max_r - min_r + 1, max_c - min_c + 1)
            })
            
        regions.sort(key=lambda x: x['size'])
        template_reg = regions[0]
        other_regions = regions[1:]
        
        marker_color = None
        for c in template_reg['colors']:
            if any(any(inp[r, col] == c for r, col in other['coords']) for other in other_regions):
                marker_color = c
                break
        if marker_color is None: return None
        
        t_marker_coords = np.array([(r, c) for r, c in template_reg['coords'] if inp[r, c] == marker_color])
        t_center = t_marker_coords.mean(axis=0).astype(int)
        t_map = {(r, c): inp[r, c] for r, c in template_reg['coords']}
        t_min_r, t_min_c, t_max_r, t_max_c = template_reg['bbox']
        t_h, t_w = t_max_r - t_min_r + 1, t_max_c - t_min_c + 1
            
        outp = inp.copy()
        for r, c in template_reg['coords']: outp[r, c] = bg
            
        for other in other_regions:
            reg_seeds = [(r, c) for r, c in other['coords'] if inp[r, c] == marker_color]
            mask = np.zeros((h, w), dtype=bool)
            for r, c in other['coords']: mask[r, c] = True
            
            for sr, sc in reg_seeds:
                for (tr, tc), color in t_map.items():
                    dr, dc = tr - t_center[0], tc - t_center[1]
                    if (dr, dc) == (0, 0): continue
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        outp[nr, nc] = color
                
                for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    dist = 1
                    bar_colors = []
                    while True:
                        tr, tc = t_center[0] + dist * direction[0], t_center[1] + dist * direction[1]
                        if not (t_min_r <= tr <= t_max_r and t_min_c <= tc <= t_max_c):
                            break
                        if (tr, tc) in t_map:
                            bar_colors.append(t_map[(tr, tc)])
                            dist += 1
                        else:
                            break
                    
                    if bar_colors:
                        last_tr, last_tc = t_center[0] + (dist - 1) * direction[0], t_center[1] + (dist - 1) * direction[1]
                        is_on_boundary = (last_tr == t_min_r or last_tr == t_max_r or last_tc == t_min_c or last_tc == t_max_c)
                        
                        if is_on_boundary:
                            if (direction[0] != 0 and t_h >= t_w) or (direction[1] != 0 and t_w >= t_h):
                                last_color = bar_colors[-1]
                                for d in range(dist, max(h, w)):
                                    nr, nc = sr + d * direction[0], sc + d * direction[1]
                                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                                        outp[nr, nc] = last_color
                                    else:
                                        break
                                
                outp[sr, sc] = marker_color
        return outp

    for inp, out in solver.pairs:
        pred = process(inp)
        if pred is None or not np.array_equal(pred, out): return None
    return [process(ti) for ti in solver.test_in]
