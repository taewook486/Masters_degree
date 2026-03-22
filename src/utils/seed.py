"""Reproducibility utilities for consistent experiment results."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value. Default experiments use 42, 123, 456.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


EXPERIMENT_SEEDS = [42, 123, 456]
