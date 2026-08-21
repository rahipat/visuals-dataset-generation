"""
Introspection SVM stage — the paper's final predictor (Paper 1, step 3).

After the two-stream CNN is trained by `baselines.train`, this:
  1. loads the best CNN checkpoint,
  2. extracts fc8 deep features on the train and val splits,
  3. fits a linear SVM regressor on the train features -> continuous failure
     fraction (the paper uses a linear SVM on CNN features instead of softmax),
  4. evaluates on val with the paper's Error-vs-Failure-Rate (Risk-Averse Metric)
     curve, failure-prediction AUROC, and a per-weather breakdown,
  5. writes a JSON report next to the checkpoint.

Run from visuals-ml/ after training:
    python -m baselines.introspection_svm --config configs/introspection.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import baselines.models  # noqa: F401  (registers the model)
from baselines.core.registry import build_model
from baselines.core.runner import _make_loader, _maybe_cap
from baselines.core.utils import load_config
from baselines.models.introspection import _auroc


def _error_vs_failure_rate(pred_score, underlying_err, grid):
    """Mean underlying error over the most-reliable retained fraction, as the
    rejected fraction (FR) sweeps `grid`. Lower/earlier = better introspection."""
    order = np.argsort(pred_score)  # ascending: most reliable (lowest score) first
    err_sorted = underlying_err[order]
    curve = []
    n = len(err_sorted)
    for fr in grid:
        keep = max(1, int(round((1.0 - fr) * n)))
        curve.append(float(err_sorted[:keep].mean()))
    return curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Defaults to <checkpoint_dir>/best.pt")
    parser.add_argument("--fail-thresh", type=float, default=0.5,
                        help="fail_frac threshold for the binary AUROC label.")
    parser.add_argument("--C", type=float, default=1.0, help="Linear SVM C.")
    args = parser.parse_args()

    try:
        from sklearn.svm import LinearSVR
    except ImportError as e:
        raise SystemExit("scikit-learn is required for the SVM stage: "
                         "pip install scikit-learn") from e

    cfg = load_config(args.config)
    checkpoint = args.checkpoint or f"{cfg['checkpoint_dir']}/best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  checkpoint: {checkpoint}")

    model = build_model(cfg)
    train_set, val_set = model.build_datasets(cfg)
    train_set, val_set = _maybe_cap(train_set, cfg), _maybe_cap(val_set, cfg)
    print(f"Train: {len(train_set)}  Val: {len(val_set)}")
    model.to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    train_loader = _make_loader(train_set, model, cfg, shuffle=False, device=device)
    val_loader = _make_loader(val_set, model, cfg, shuffle=False, device=device)

    xtr, ytr, _etr, _wtr = model.extract_fc8(train_loader, device)
    xva, yva, eva, wva = model.extract_fc8(val_loader, device)
    xtr, ytr = xtr.numpy(), ytr.numpy()
    xva, yva, eva = xva.numpy(), yva.numpy(), eva.numpy()

    svm = LinearSVR(C=args.C, max_iter=10000)
    svm.fit(xtr, ytr)
    pred = np.clip(svm.predict(xva), 0.0, 1.0)   # failure score in [0, 1]

    fail_label = (yva > args.fail_thresh).astype(int)
    grid = [i / 10 for i in range(10)]           # FR = 0.0 .. 0.9
    report = {
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "auroc": _auroc(fail_label, pred),
        "efr_grid": grid,
        "efr_introspection": _error_vs_failure_rate(pred, eva, grid),
        "efr_ideal": _error_vs_failure_rate(eva, eva, grid),      # oracle upper bound
        "efr_random": _error_vs_failure_rate(np.zeros_like(eva), eva, grid),
    }

    per_weather = {}
    for w in sorted(set(wva)):
        m = np.array([x == w for x in wva])
        per_weather[w] = {
            "n": int(m.sum()),
            "mean_pred_fail": float(pred[m].mean()),
            "mean_underlying_err": float(eva[m].mean()),
        }
    report["per_weather"] = per_weather

    out = Path(checkpoint).with_name("introspection_svm_report.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nAUROC (failure prediction): {report['auroc']:.4f}")
    print(f"EFR @FR=0.0/0.3/0.5: "
          f"{report['efr_introspection'][0]:.3f} / "
          f"{report['efr_introspection'][3]:.3f} / "
          f"{report['efr_introspection'][5]:.3f} m")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
