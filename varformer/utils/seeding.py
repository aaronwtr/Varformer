"""Random seed utilities."""
import random

import numpy as np
import torch


class random_seed_context:
    """Context manager that temporarily sets the Python random seed."""

    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = random.getstate()
        random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        random.setstate(self.state)


def set_seed(seed: int) -> None:
    """Seed PyTorch, NumPy, and Python random for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
