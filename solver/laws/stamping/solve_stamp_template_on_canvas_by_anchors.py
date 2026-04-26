import numpy as np
from typing import List, Optional

def solve_stamp_template_on_canvas_by_anchors(solver) -> Optional[List[np.ndarray]]:
    """
    Splits the grid into two halves (Horizontal or Vertical).
    Identifies one half as the 'canvas' and the other as the 'template'.
    Finds 'anchor' colors that exist in both canvas and template.
    Matches template components to canvas positions using shared anchor points.
    Stamps the template components onto the canvas.
    """
    def get_components(grid, background_color):
        h, w = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        components = []
        for r in range(h):
            for c in range(w):
                if not visited[r, c] and grid[r, c] != background_color:
                    comp = []
                    stack = [(r, c)]
                    visited[r, c] = True
                    while stack:
                        curr_r, curr_c = stack.pop()
                        comp.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] != background_color:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                    components.append(comp)
        return components

    def apply_logic(input_grid):
        inp = np.array(input_grid)
        h_in, w_in = inp.shape
        
        possible_splits = []
        if h_in % 2 == 0: possible_splits.append((inp[0:h_in//2, :], inp[h_in//2:h_in, :]))
        if w_in % 2 == 0: possible_splits.append((inp[:, 0:w_in//2], inp[:, w_in//2:w_in]))

        for reg0, reg1 in possible_splits:
            for canvas, template in [(reg0, reg1), (reg1, reg0)]:
                u_c, counts_c = np.unique(canvas, return_counts=True)
                if len(counts_c) == 0: continue
                canvas_bg = u_c[np.argmax(counts_c)]
                
                u_t, counts_t = np.unique(template, return_counts=True)
                if len(counts_t) == 0: continue
                template_bg = u_t[np.argmax(counts_t)]
                
                canvas_anchor_colors = set()
                unmatched_canvas_anchors = set()
                for r in range(canvas.shape[0]):
                    for c in range(canvas.shape[1]):
                        if canvas[r, c] != canvas_bg:
                            unmatched_canvas_anchors.add((r, c, canvas[r, c]))
                            canvas_anchor_colors.add(canvas[r, c])
                
                if not unmatched_canvas_anchors: continue

                t_comps = get_components(template, template_bg)
                comps_with_anchors = []
                for comp in t_comps:
                    c_anchors = [(r, c, template[r, c]) for r, c in comp if template[r, c] in canvas_anchor_colors]
                    comps_with_anchors.append((comp, c_anchors))
                
                comps_with_anchors.sort(key=lambda x: len(x[1]), reverse=True)
                
                output = canvas.copy()
                any_placed = False
                for comp, comp_anchors in comps_with_anchors:
                    if not comp_anchors: continue
                    found_shift = None
                    first_r, first_c, first_color = comp_anchors[0]
                    for ar, ac, a_color in list(unmatched_canvas_anchors):
                        if a_color == first_color:
                            dr, dc = ar - first_r, ac - first_c
                            match = True
                            translated_anchors = []
                            for cr, cc, c_color in comp_anchors:
                                nr, nc = cr + dr, cc + dc
                                tgt = (nr, nc, c_color)
                                if tgt not in unmatched_canvas_anchors:
                                    match = False; break
                                translated_anchors.append(tgt)
                            if match:
                                found_shift = (dr, dc)
                                for tgt in translated_anchors: unmatched_canvas_anchors.remove(tgt)
                                break
                    if found_shift:
                        dr, dc = found_shift
                        for r, c in comp:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < output.shape[0] and 0 <= nc < output.shape[1]:
                                output[nr, nc] = template[r, c]
                        any_placed = True
                if any_placed and not np.array_equal(output, canvas):
                    return output
        return None

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
