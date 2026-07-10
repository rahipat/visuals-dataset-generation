"""Shared helpers used across baselines."""

import random

import yaml
from torch.utils.data import Subset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def split_dataset(dataset, val_fraction: float, seed: int):
    """Deterministic train/val split by shuffling indices with a fixed seed.

    Matches the convention used by the original visuals-ml train.py/eval.py so a
    model re-registered under the harness sees the same split it always did.
    """
    n = len(dataset)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    split = int(n * (1 - val_fraction))
    return Subset(dataset, indices[:split]), Subset(dataset, indices[split:])
