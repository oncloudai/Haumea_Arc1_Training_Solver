import numpy as np

def solve_grid_cell_template_growth(solver):
    def get_grid_structure(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        best_color = -1
        max_lines = -1
        best_rows = []
        best_cols = []
        for color in range(1, 10):
            row_lines = [r for r in range(rows) if np.all(grid[r, :] == color)]
            col_lines = [c for c in range(cols) if np.all(grid[:, c] == color)]
            if len(row_lines) + len(col_lines) > max_lines:
                max_lines = len(row_lines) + len(col_lines)
                best_color = color
                best_rows = row_lines
                best_cols = col_lines
        if max_lines > 0:
            return best_color, best_rows, best_cols
        return None, [], []

    def get_cell_matrix(grid, gc, r_lines, c_lines):
        r_boundaries = [-1] + r_lines + [grid.shape[0]]
        c_boundaries = [-1] + c_lines + [grid.shape[1]]
        cell_matrix = []
        for r_idx in range(len(r_boundaries) - 1):
            row_cells = []
            for c_idx in range(len(c_boundaries) - 1):
                r_start = r_boundaries[r_idx] + 1
                r_end = r_boundaries[r_idx+1]
                c_start = c_boundaries[c_idx] + 1
                c_end = c_boundaries[c_idx+1]
                if r_start < r_end and c_start < c_end:
                    cell = grid[r_start:r_end, c_start:c_end]
                    counts = np.bincount(cell.flatten(), minlength=10)
                    row_cells.append(np.argmax(counts))
            if row_cells:
                cell_matrix.append(row_cells)
        return np.array(cell_matrix), r_boundaries, c_boundaries

    def apply_to_grid(grid, cell_matrix, r_boundaries, c_boundaries):
        new_grid = grid.copy()
        for r_idx in range(cell_matrix.shape[0]):
            for c_idx in range(cell_matrix.shape[1]):
                r_start = r_boundaries[r_idx] + 1
                r_end = r_boundaries[r_idx+1]
                c_start = c_boundaries[c_idx] + 1
                c_end = c_boundaries[c_idx+1]
                if r_start < r_end and c_start < c_end:
                    new_grid[r_start:r_end, c_start:c_end] = cell_matrix[r_idx, c_idx]
        return new_grid

    def get_pattern(inp_cells, out_cells, color):
        rows, cols = inp_cells.shape
        coords = np.argwhere(inp_cells == color)
        if len(coords) == 0: return None
        pattern = {}
        for r, c in coords:
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if out_cells[nr, nc] != 0 and out_cells[nr, nc] != color:
                            pattern[(dr, dc)] = out_cells[nr, nc]
        return pattern

    # Try each color as the potential center
    for center_color in range(1, 10):
        # Collect potential pattern from all training examples
        global_pattern = {}
        consistent = True
        found_center = False
        
        for inp, out in solver.pairs:
            gc, rl, cl = get_grid_structure(inp)
            if gc is None: consistent = False; break
            in_cells, _, _ = get_cell_matrix(inp, gc, rl, cl)
            out_cells_target, _, _ = get_cell_matrix(out, gc, rl, cl)
            
            if np.any(in_cells == center_color):
                found_center = True
                p = get_pattern(in_cells, out_cells_target, center_color)
                for k, v in p.items():
                    if k in global_pattern and global_pattern[k] != v:
                        consistent = False; break
                    global_pattern[k] = v
                if not consistent: break
            
            # Even if not found in one example, it might be in others.
            # But we must check if applying the currently known pattern matches the output.
            # Wait, we need to collect the full pattern first.
        
        if not consistent or not found_center: continue
        
        # Now verify the global_pattern against all training examples
        all_match = True
        for inp, out in solver.pairs:
            gc, rl, cl = get_grid_structure(inp)
            in_cells, _, _ = get_cell_matrix(inp, gc, rl, cl)
            out_cells_target, _, _ = get_cell_matrix(out, gc, rl, cl)
            
            pred_cells = in_cells.copy()
            coords = np.argwhere(in_cells == center_color)
            for r, c in coords:
                for (dr, dc), val in global_pattern.items():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < in_cells.shape[0] and 0 <= nc < in_cells.shape[1]:
                        pred_cells[nr, nc] = val
            
            if not np.array_equal(pred_cells, out_cells_target):
                all_match = False; break
        
        if all_match:
            # Successfully found the pattern! Apply to test cases.
            results = []
            for ti in solver.test_in:
                gc, rl, cl = get_grid_structure(ti)
                if gc is None: results.append(ti.copy()); continue
                ti_cells, rb, cb = get_cell_matrix(ti, gc, rl, cl)
                
                pred_cells = ti_cells.copy()
                coords = np.argwhere(ti_cells == center_color)
                for r, c in coords:
                    for (dr, dc), val in global_pattern.items():
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < ti_cells.shape[0] and 0 <= nc < ti_cells.shape[1]:
                            pred_cells[nr, nc] = val
                results.append(apply_to_grid(ti, pred_cells, rb, cb))
            return results
            
    return None
