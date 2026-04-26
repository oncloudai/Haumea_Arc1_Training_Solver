import numpy as np
from typing import List, Optional, Tuple, Callable

class MegaSolver:
    def __init__(self, data: dict):
        self.train_in = [np.array(x['input']) for x in data['train']]
        self.train_out = [np.array(x['output']) for x in data['train']]
        self.test_in = [np.array(x['input']) for x in data['test']]
        self.pairs = list(zip(self.train_in, self.train_out))
        self.methods: List[Callable] = []

    def register_method(self, method: Callable):
        self.methods.append(method)

    def brute_force(self) -> Tuple[Optional[List[np.ndarray]], str]:
        for method in self.methods:
            try:
                preds = method(self)
                if preds is not None:
                    return preds, method.__name__
            except Exception as e:
                continue
        return None, "Unsolved"
