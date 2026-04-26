import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_fill_frames_and_shoot_beams(solver) -> Optional[List[np.ndarray]]:
    """
    1. Identify all color bounding boxes and their boundaries.
    2. Fill the largest bounding box with its color.
    3. Identify beam sources: single pixels of a color on its bbox boundary.
    4. Vertical beams from top/bottom sources fill non-boundary pixels.
    5. Horizontal beams from left/right sources fill pixels outside all bboxes.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        color_infos = {}
        all_bboxes = []
        all_boundaries = set()
        
        for color in np.unique(grid):
            if color == 0: continue
            mask = (grid == color).astype(int)
            labeled, num_f = label(mask)
            if num_f == 0: continue
            
            # Largest component
            main_idx = np.argmax([np.sum(labeled == j) for j in range(1, num_f + 1)]) + 1
            main_mask = (labeled == main_idx)
            rows, cols = np.where(main_mask)
            r1, r2, c1, c2 = rows.min(), rows.max(), cols.min(), cols.max()
            
            sources = []
            # Top edge
            top_pix = [(r1, c) for c in range(c1, c2 + 1) if main_mask[r1, c]]
            if len(top_pix) == 1: sources.append((top_pix[0], 'v'))
            # Bottom edge
            bot_pix = [(r2, c) for c in range(c1, c2 + 1) if main_mask[r2, c]]
            if len(bot_pix) == 1: sources.append((bot_pix[0], 'v'))
            # Left edge
            left_pix = [(r, c1) for r in range(r1, r2 + 1) if main_mask[r, c1]]
            if len(left_pix) == 1: sources.append((left_pix[0], 'h'))
            # Right edge
            right_pix = [(r, c2) for r in range(r1, r2 + 1) if main_mask[r, c2]]
            if len(right_pix) == 1: sources.append((right_pix[0], 'h'))
            
            boundary = set([(r, c) for r in range(r1, r2 + 1) for c in [c1, c2]] + 
                           [(r, c) for r in [r1, r2] for c in range(c1, c2 + 1)])
            
            color_infos[color] = {
                'bbox': (r1, r2, c1, c2),
                'sources': sources,
                'area': (r2 - r1 + 1) * (c2 - c1 + 1)
            }
            all_bboxes.append((r1, r2, c1, c2))
            all_boundaries.update(boundary)
            
        # 1. Start output
        out = grid.copy()
        
        # 2. Fill largest bbox
        if color_infos:
            largest_color = max(color_infos.keys(), key=lambda c: color_infos[c]['area'])
            r1, r2, c1, c2 = color_infos[largest_color]['bbox']
            out[r1:r2+1, c1:c2+1] = largest_color
            
        # 3. Draw beams
        final_out = out.copy()
        for color, info in color_infos.items():
            for (sr, sc), b_type in info['sources']:
                if b_type == 'v':
                    # Vertical beam fills non-boundary pixels
                    for r in range(h):
                        if (r, sc) not in all_boundaries:
                            final_out[r, sc] = color
                else:
                    # Horizontal beam fills pixels outside ALL bboxes
                    for c in range(w):
                        is_inside = False
                        for (br1, br2, bc1, bc2) in all_bboxes:
                            if br1 <= sr <= br2 and bc1 <= c <= bc2:
                                is_inside = True; break
                        if not is_inside:
                            final_out[sr, c] = color
                            
        return final_out

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
