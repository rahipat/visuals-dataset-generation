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
from baselines.core.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # cfg['seed'] previously only seeded the train/val split (a local
    # random.Random in core/utils.split_dataset); torch's global RNG was left
    # unseeded, so weight init AND DataLoader shuffle order differed on every
    # run. That makes a divergence impossible to reproduce or bisect -- two
    # runs can't be compared when neither the starting weights nor the batch
    # order match. Seed torch here, before build_model() initialises weights.
    seed = cfg.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Seeded torch RNG with {seed} (weight init + shuffle order now "
              "reproducible across runs)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  model: {cfg['model']}")

    model = build_model(cfg)
    train(model, cfg, device, resume=args.resume)


if __name__ == "__main__":
    main()
