"""
Harness evaluation entrypoint. Reports the model's metrics, including the
per-weather-variant breakdown.

Run from the visuals-ml/ directory:
    python -m baselines.eval --config configs/positionnet.yaml
"""

import argparse
import json

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import baselines.models  # noqa: F401  (registers all baselines)
from baselines.core.registry import build_model
from baselines.core.runner import evaluate
from baselines.core.utils import apply_index_suffix, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Defaults to <checkpoint_dir>/best.pt")
    parser.add_argument("--index-suffix", default=None,
                        help="Insert '_<suffix>' before .jsonl in index_file. "
                             "Use the all-weather index here: weather is the "
                             "test axis, so evaluation must span every variant.")
    parser.add_argument("--report", default=None,
                        help="Write the full metrics dict to this JSON path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_index_suffix(cfg, args.index_suffix)
    checkpoint = args.checkpoint or f"{cfg['checkpoint_dir']}/best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  model: {cfg['model']}")

    model = build_model(cfg)
    metrics = evaluate(model, cfg, device, checkpoint)

    if args.report:
        from pathlib import Path
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"model": cfg["model"], "config": args.config,
                       "checkpoint": checkpoint,
                       "index_file": cfg.get("index_file"),
                       "metrics": metrics}, f, indent=2, default=str)
        print(f"Wrote report to {out}")

    per_weather = metrics.pop("per_weather", None)
    print("\n=== Overall ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if per_weather:
        print("\n=== Per weather variant ===")
        print(json.dumps(per_weather, indent=2))


if __name__ == "__main__":
    main()
