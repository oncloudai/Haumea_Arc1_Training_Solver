import numpy as np
from typing import List, Optional

def solve_fill_object_holes_with_matched_patterns(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies the largest non-background object (frame color) and its internal 'pockets' (holes).
    Extracts small 3x3 pattern objects from elsewhere in the grid.
    Finds which pattern (including rotations/flips) matches each pocket perfectly.
    Fills each pocket with its corresponding matched pattern.
    """
    def label_components(mask, connectivity=8):
        rows, cols = mask.shape
        labeled = np.zeros((rows, cols), dtype=int)
        label_count = 0
        for r in range(rows):
            for c in range(cols):
                if mask[r, c] and labeled[r, c] == 0:
                    label_count += 1
                    q = [(r, c)]
                    labeled[r, c] = label_count
                    while q:
                        curr_r, curr_c = q.pop(0)
                        neighbors = []
                        if connectivity == 8:
                            for dr in [-1, 0, 1]:
                                for dc in [-1, 0, 1]:
                                    if dr == 0 and dc == 0: continue
                                    neighbors.append((curr_r + dr, curr_c + dc))
                        else:
                            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                neighbors.append((curr_r + dr, curr_c + dc))
                        for nr, nc in neighbors:
                            if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and labeled[nr, nc] == 0:
                                labeled[nr, nc] = label_count
                                q.append((nr, nc))
        return labeled, label_count

    def get_symmetries(grid):
        soms = []
        curr = np.array(grid)
        for _ in range(4):
            soms.append(curr.copy())
            soms.append(np.fliplr(curr).copy())
            curr = np.rot90(curr)
        unique_soms = []
        seen = set()
        for s in soms:
            t = tuple(s.flatten())
            if t not in seen:
                unique_soms.append(s)
                seen.add(t)
        return unique_soms

    def apply_logic(input_grid):
        input_grid = np.array(input_grid)
        rows, cols = input_grid.shape
        unique, counts = np.unique(input_grid, return_counts=True)
        # Background is usually the most common color
        bg_color = int(unique[np.argmax(counts)])
        
        # Try different potential "frame" colors
        other_colors = [int(c) for c in unique if c != bg_color]
        if not other_colors: return input_grid
        
        # Heuristic: the frame is the largest connected component of a non-bg color
        best_frame_info = None
        max_size = -1
        for fc in other_colors:
            fc_mask = (input_grid == fc)
            labeled_fc, num_fc = label_components(fc_mask, connectivity=8)
            for i in range(1, num_fc + 1):
                size = np.sum(labeled_fc == i)
                if size > max_size:
                    max_size = size
                    best_frame_info = (fc, labeled_fc == i)
        
        if not best_frame_info: return input_grid
        frame_color, frame_mask = best_frame_info
        
        frame_pixels = np.argwhere(frame_mask)
        min_r, min_c = frame_pixels.min(axis=0)
        max_r, max_c = frame_pixels.max(axis=0)
        fr_h, fr_w = max_r - min_r + 1, max_c - min_c + 1
        
        frame_crop = np.full((fr_h, fr_w), bg_color)
        for pr, pc in frame_pixels:
            frame_crop[pr - min_r, pc - min_c] = frame_color
            
        # Pockets are holes in the frame crop
        hole_mask = (frame_crop == bg_color)
        labeled_holes, num_pockets = label_components(hole_mask, connectivity=8)
        pockets = []
        for i in range(1, num_pockets + 1):
            p_holes = set(tuple(p) for p in np.argwhere(labeled_holes == i))
            pockets.append(p_holes)
            
        # Patterns are 3x3 (or other small size) objects outside the frame
        non_bg_mask = (input_grid != bg_color)
        labeled_all, num_all_features = label_components(non_bg_mask, connectivity=8)
        patterns = []
        for i in range(1, num_all_features + 1):
            comp_pixels = np.argwhere(labeled_all == i)
            # If any pixel of this component is part of our frame, skip it
            if any(frame_mask[pr, pc] for pr, pc in comp_pixels): continue
            
            p_min_r, p_min_c = comp_pixels.min(axis=0)
            p_max_r, p_max_c = comp_pixels.max(axis=0)
            ph, pw = p_max_r - p_min_r + 1, p_max_c - p_min_c + 1
            
            # Usually 3x3 patterns are embedded in a 3x3 region
            pr1 = max(0, min(rows - 3, p_min_r))
            pc1 = max(0, min(cols - 3, p_min_c))
            patterns.append(input_grid[pr1:pr1+3, pc1:pc1+3].copy())

        # For each pattern, find all possible placements in the frame holes
        pattern_placements = []
        for p in patterns:
            placements = []
            for sym in get_symmetries(p):
                # Try all offsets within the frame crop
                sym_mask = (sym != bg_color)
                sym_frame_mask = (sym == frame_color)
                sym_other_mask = (sym != bg_color) & (sym != frame_color)
                
                for r in range(fr_h - 2):
                    for c in range(fr_w - 2):
                        # Mode A: Pattern's frame color (2) matches the frame's holes
                        match_a = True
                        for dr in range(3):
                            for dc in range(3):
                                is_hole = (frame_crop[r+dr, c+dc] == bg_color)
                                if sym_frame_mask[dr, dc]:
                                    if not is_hole: match_a = False; break
                                else:
                                    if is_hole: match_a = False; break
                            if not match_a: break
                        if match_a:
                            perim = 0
                            for dr in range(3):
                                for dc in range(3):
                                    if sym[dr, dc] != frame_color:
                                        rr, cc = r+dr, c+dc
                                        if rr == 0 or rr == fr_h-1 or cc == 0 or cc == fr_w-1:
                                            perim += 1
                            placements.append({'r': r, 'c': c, 'sym': sym, 'mode': 'A', 'perim': perim})
                            
                        # Mode B: Pattern's colored pixels match the frame's holes
                        match_b = True
                        for dr in range(3):
                            for dc in range(3):
                                is_hole = (frame_crop[r+dr, c+dc] == bg_color)
                                if sym_other_mask[dr, dc]:
                                    if not is_hole: match_b = False; break
                                else:
                                    if is_hole: match_b = False; break
                            if not match_b: break
                        if match_b:
                            placements.append({'r': r, 'c': c, 'sym': sym, 'mode': 'B', 'perim': 0})
            pattern_placements.append(placements)

        final_placements = []
        used_patterns = [False] * len(patterns)
        for pocket in pockets:
            best_p_idx = None
            best_p = None
            for p_idx, placements in enumerate(pattern_placements):
                if used_patterns[p_idx]: continue
                for p in placements:
                    if p['mode'] == 'A':
                        p_holes = set((p['r']+dr, p['c']+dc) for dr in range(3) for dc in range(3) if p['sym'][dr, dc] == frame_color)
                    else:
                        p_holes = set((p['r']+dr, p['c']+dc) for dr in range(3) for dc in range(3) if p['sym'][dr, dc] != bg_color and p['sym'][dr, dc] != frame_color)
                    
                    if p_holes == pocket:
                        if best_p is None or p['perim'] < best_p['perim']:
                            best_p = p
                            best_p_idx = p_idx
            if best_p_idx is not None:
                final_placements.append(best_p)
                used_patterns[best_p_idx] = True
        
        output_crop = frame_crop.copy()
        for p in final_placements:
            for dr in range(3):
                for dc in range(3):
                    val = int(p['sym'][dr, dc])
                    if val != bg_color:
                        output_crop[p['r']+dr, p['c']+dc] = val
        return output_crop

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out_expected.shape or not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
