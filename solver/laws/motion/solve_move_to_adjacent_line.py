
import numpy as np

def solve_move_to_adjacent_line(solver):
    def apply_logic(inp):
        inp = np.array(inp)
        h, w = inp.shape
        outp = np.zeros_like(inp)
        
        # Find vertical lines
        v_lines = {} # color -> [cols]
        for c in range(w):
            col = inp[:, c]
            unique, counts = np.unique(col, return_counts=True)
            for i in range(len(unique)):
                if unique[i] != 0 and counts[i] == h:
                    color = unique[i]
                    if color not in v_lines: v_lines[color] = []
                    v_lines[color].append(c)
                        
        # Find horizontal lines
        h_lines = {} # color -> [rows]
        for r in range(h):
            row = inp[r, :]
            unique, counts = np.unique(row, return_counts=True)
            for i in range(len(unique)):
                if unique[i] != 0 and counts[i] == w:
                    color = unique[i]
                    if color not in h_lines: h_lines[color] = []
                    h_lines[color].append(r)

        if not v_lines and not h_lines:
            return None

        # Re-draw the lines
        for color, cols in v_lines.items():
            for c in cols:
                outp[:, c] = color
        for color, rows in h_lines.items():
            for r in rows:
                outp[r, :] = color

        # Move other pixels of line colors
        for r in range(h):
            for c in range(w):
                color = inp[r, c]
                if color == 0: continue
                
                moved = False
                # Check if it belongs to a vertical line color
                if color in v_lines:
                    cols = v_lines[color]
                    if c not in cols:
                        best_lc = min(cols, key=lambda lc: abs(lc - c))
                        target_c = best_lc + 1 if c > best_lc else best_lc - 1
                        if 0 <= target_c < w:
                            outp[r, target_c] = color
                            moved = True
                
                # Check if it belongs to a horizontal line color
                if not moved and color in h_lines:
                    rows = h_lines[color]
                    if r not in rows:
                        best_lr = min(rows, key=lambda lr: abs(lr - r))
                        target_r = best_lr + 1 if r > best_lr else best_lr - 1
                        if 0 <= target_r < h:
                            outp[target_r, c] = color
                            moved = True
        return outp

    results = []
    # Verification
    for inp, outp in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, outp):
            return None
            
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
