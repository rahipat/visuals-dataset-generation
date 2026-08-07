"""
Harness training entrypoint. Model-agnostic — picks the baseline from the
config's `model:` key.

Run from the visuals-ml/ directory:
    python -m baselines.train --config configs/positionnet.yaml
"""

import argparse

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import baselines.models  # noqa: F401  (registers all baselines)
from baselines.core.registry import build_model
from baselines.core.runner import train
from baselines.core.utils import apply_index_suffix, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from.")
    parser.add_argument("--index-suffix", default=None,
                        help="Insert '_<suffix>' before .jsonl in index_file, so "
                             "smoke/full/full-eval indexes coexist without "
                             "separate configs.")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Override the config's checkpoint_dir.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_index_suffix(cfg, args.index_suffix)
    if args.checkpoint_dir:
        cfg["checkpoint_dir"] = args.checkpoint_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  model: {cfg['model']}")

    model = build_model(cfg)
    train(model, cfg, device, resume=args.resume)


if __name__ == "__main__":
    main()
