# core/utils/seed.py
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """
    Fix random seeds for reproducibility.

    deterministic=False (default):
      - faster, but results may vary slightly due to CUDA kernels

    deterministic=True:
      - more reproducible, may be slower
    """
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Optional: make matmul deterministic-ish (PyTorch version dependent)
        # torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
