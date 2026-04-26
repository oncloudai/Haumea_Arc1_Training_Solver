import numpy as np
from typing import List, Optional

def solve_mask_intersection(solver) -> Optional[List[np.ndarray]]:
    if len(solver.train_in) < 2: return None
    shape = solver.train_in[0].shape
    if not all(inp.shape == shape for inp in solver.train_in): return None
    return None
