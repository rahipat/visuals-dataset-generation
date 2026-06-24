"""
The BaselineModel contract — the single seam every baseline implements.

The runner (core/runner.py) owns everything model-independent: the epoch loop,
optimizer, AMP, checkpointing, logging. Each baseline owns the parts that differ
between a per-object regressor (PositionNet) and a per-image set-prediction
detector (MonoDETR):

  - build_datasets : how raw records become (train, val) Datasets
  - collate_fn     : how samples batch together (variable-length for detection)
  - training_step  : forward + loss for one batch  -> (scalar_loss, log_dict)
  - evaluate       : the whole eval pass           -> metrics dict

Convention: evaluate() MUST return a dict containing a float key "monitor" where
LOWER IS BETTER (e.g. mean error; for higher-is-better metrics like AP, return
the negative). The runner checkpoints whenever "monitor" improves. All other keys
are logged as-is.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate


class BaselineModel(nn.Module, ABC):
    name: str = "base"

    @classmethod
    def from_config(cls, cfg: dict) -> "BaselineModel":
        """Build a model instance from a config dict. Override if construction
        needs more than passing cfg to __init__."""
        return cls(cfg)

    @staticmethod
    @abstractmethod
    def build_datasets(cfg: dict):
        """Return (train_dataset, val_dataset)."""
        raise NotImplementedError

    def collate_fn(self, batch):
        """Default to torch's collate; detection models override this."""
        return default_collate(batch)

    @abstractmethod
    def training_step(self, batch, device: torch.device):
        """Run forward + loss for one batch. Called inside the runner's autocast
        context, so do NOT call backward here. Return (loss, log_dict) where
        loss is a scalar tensor and log_dict maps names -> python floats."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, loader, device: torch.device) -> dict:
        """Run the full validation pass and return a metrics dict containing a
        float 'monitor' key (lower is better). Implementations should wrap their
        loop in torch.no_grad()."""
        raise NotImplementedError
