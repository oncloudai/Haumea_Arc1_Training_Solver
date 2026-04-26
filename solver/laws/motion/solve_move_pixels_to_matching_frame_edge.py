import numpy as np
from typing import List, Optional

def solve_move_pixels_to_matching_frame_edge(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies colors on the four edges of the grid frame.
    Moves interior non-zero pixels to the row or column adjacent to the edge that matches their color.
    Interior pixels whose color doesn't match any edge color are removed.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        if rows < 3 or cols < 3: return None
        
        # Identify side colors from the frame (excluding corners which are often 0)
        top_color = grid[0, cols // 2]
        bottom_color = grid[rows-1, cols // 2]
        left_color = grid[rows // 2, 0]
        right_color = grid[rows // 2, cols-1]
        
        # Create output grid with the same frame
        output = np.zeros_like(grid)
        output[0, :] = grid[0, :]
        output[rows-1, :] = grid[rows-1, :]
        output[:, 0] = grid[:, 0]
        output[:, cols-1] = grid[:, cols-1]
        
        # Process interior seeds
        found_any = False
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                color = grid[r, c]
                if color != 0:
                    found_any = True
                    # Move to the edge with matching color (adjacent to the frame)
                    if color == top_color:
                        output[1, c] = color
                    elif color == bottom_color:
                        output[rows - 2, c] = color
                    elif color == left_color:
                        output[r, 1] = color
                    elif color == right_color:
                        output[r, cols - 2] = color
        
        if not found_any: return None
        return output

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
