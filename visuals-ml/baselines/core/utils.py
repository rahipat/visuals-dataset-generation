"""Shared helpers used across baselines."""

from __future__ import annotations

import random
from pathlib import Path

import yaml
from torch.utils.data import Subset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_index_suffix(cfg: dict, suffix: str | None) -> dict:
    """Rewrite cfg['index_file'] to carry a suffix before its extension.

    Lets one config serve the smoke / full / full-eval indexes:
        data/output/det_records.jsonl -> data/output/det_records_full.jsonl
    Returns cfg unchanged when suffix is falsy.
    """
    if not suffix or not cfg.get("index_file"):
        return cfg
    p = Path(cfg["index_file"])
    cfg = dict(cfg)
    cfg["index_file"] = str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))
    print(f"index_file -> {cfg['index_file']}")
    return cfg


def split_dataset(dataset, val_fraction: float, seed: int):
    """Deterministic train/val split by shuffling indices with a fixed seed.

    Matches the convention used by the original visuals-ml train.py/eval.py so a
    model re-registered under the harness sees the same split it always did.

    WARNING: this splits on individual records and therefore LEAKS on the visuals
    dataset, where each physical object appears once per weather variant (10x)
    and again in every neighbouring ~10 Hz frame. Use split_by_group() with the
    segment id for any number you intend to report.
    """
    n = len(dataset)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    split = int(n * (1 - val_fraction))
    return Subset(dataset, indices[:split]), Subset(dataset, indices[split:])


def split_by_group(dataset, groups, val_fraction: float, seed: int):
    """Train/val split that keeps every sample of a group on one side.

    `groups[i]` is the group key of sample i (use the segment id). Whole groups
    are assigned to val until the val fraction is reached, so a frame's clear /
    rain / fog renderings — and its temporal neighbours — can never straddle the
    split. Group sizes differ, so the realised fraction is approximate.

    Returns (train_subset, val_subset).
    """
    if len(groups) != len(dataset):
        raise ValueError(
            f"groups has {len(groups)} entries but dataset has {len(dataset)}"
        )

    by_group = {}
    for idx, key in enumerate(groups):
        by_group.setdefault(key, []).append(idx)

    keys = sorted(by_group)
    random.Random(seed).shuffle(keys)

    target_val = len(dataset) * val_fraction
    val_idx, train_idx, n_val = [], [], 0
    for key in keys:
        members = by_group[key]
        if n_val < target_val:
            val_idx.extend(members)
            n_val += len(members)
        else:
            train_idx.extend(members)

    if not train_idx or not val_idx:
        raise ValueError(
            f"Group split degenerate: {len(keys)} group(s) gave "
            f"{len(train_idx)} train / {len(val_idx)} val samples. "
            "Need at least 2 groups (segments) to split on."
        )

    train_idx.sort()
    val_idx.sort()
    print(f"Group split: {len(keys)} groups -> "
          f"{len(train_idx)} train / {len(val_idx)} val samples "
          f"({len(val_idx) / len(dataset):.1%} val)")
    return Subset(dataset, train_idx), Subset(dataset, val_idx)
