"""
Failure-label pre-pass for the Introspective Perception baseline (Paper 1).

Paper 1 is system-agnostic: it learns to predict, from the input alone, whether
an underlying perception system will fail on a frame. That requires a supervised
target y_i = the underlying system's per-frame (in)accuracy. Here the underlying
system is PositionNet (the 3D-position regressor already in this repo), matching
the project decision to treat it as the fixed substrate whose failures we score.

This stage reads the model-free index from build_introspection_index.py, runs a
trained PositionNet over each matched object, and writes an augmented index with
two per-frame labels added:

  "fail_frac"  in [0,1]  fraction of the frame's objects whose 3D-centre error
                         exceeds --tau metres  (the continuous SVM regression
                         target; higher == less reliable, per the paper)
  "fail"       0/1        1 if fail_frac > --fail-thresh  (the binary target the
                         two-stream CNN is trained on with softmax loss)
  "mean_err"   metres     mean 3D-centre error over the frame's objects (logging)

Mirrors Paper 1's "fraction of trajectories correctly predicted" target, adapted
to per-object 3D-position error.

Usage (from visuals-ml/):
    python -m baselines.data.introspection_label_prepass \
        --index-file data/output/introspection_records.jsonl \
        --checkpoint baselines/checkpoints/positionnet/best.pt \
        --out-file data/output/introspection_labeled.jsonl \
        --tau 2.0 --fail-thresh 0.5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.paths import to_posix
from data.dataset import (
    INPUT_KEYS, _crop_box_for, _to_crop_tensor, _to_full_tensor,
)
from model.position_net import PositionNet


def _load_positionnet(checkpoint: str, device) -> PositionNet:
    """Load a PositionNet core from either a harness checkpoint (keys prefixed
    'core.') or a bare PositionNet state dict."""
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    core = {}
    for k, v in state.items():
        if k.startswith("core."):
            core[k[len("core."):]] = v
        elif k.startswith("encoder_") or k.startswith("head") or k.startswith("pool"):
            core[k] = v
    if not core:
        raise KeyError(
            f"No PositionNet weights found in {checkpoint}. "
            "Expected keys under 'core.' or a bare PositionNet state dict."
        )
    model = PositionNet()
    model.load_state_dict(core)
    return model.to(device).eval()


def _frame_inputs(record, device):
    """Build PositionNet inputs for every object in a frame. The full image is
    shared; crops/coords/targets are per object."""
    img = Image.open(to_posix(record["image_path"])).convert("RGB")
    w, h = img.size
    full = _to_full_tensor(img)

    intr = record["intrinsic"]
    crops, coords, targets = [], [], []
    for o in record["objects"]:
        box = _crop_box_for(w, h, o["cx_n"], o["cy_n"], o["sw_n"], o["sh_n"])
        crops.append(_to_crop_tensor(img.crop(box)))
        feat = {**o, "fu": intr["fu"], "fv": intr["fv"],
                "cu": intr["cu"], "cv": intr["cv"]}
        coords.append(torch.tensor([feat[k] for k in INPUT_KEYS], dtype=torch.float32))
        targets.append(torch.tensor([o["tx"], o["ty"], o["tz"]], dtype=torch.float32))

    n = len(crops)
    full = full.unsqueeze(0).expand(n, -1, -1, -1).to(device)
    crop = torch.stack(crops).to(device)
    coord = torch.stack(coords).to(device)
    target = torch.stack(targets).to(device)
    return full, crop, coord, target


@torch.no_grad()
def label_index(index_file, checkpoint, out_file, tau, fail_thresh, device,
                tau_percentile=None):
    model = _load_positionnet(checkpoint, device)
    records = [json.loads(l) for l in open(index_file, encoding="utf-8") if l.strip()]

    # Pass 1: PositionNet error per object per frame.
    frame_errs = []
    for r in records:
        full, crop, coord, target = _frame_inputs(r, device)
        err = (model(full, crop, coord) - target).norm(dim=1)   # (n,) metres
        frame_errs.append(err.cpu())

    # A fixed tau saturates when the base model is uniformly weak (all frames
    # become failures, leaving the CNN softmax with no negative class). Deriving
    # tau from a percentile of the per-object error distribution guarantees a
    # non-degenerate split, and is how tau should be calibrated against the real
    # trained PositionNet rather than guessed in absolute metres.
    if tau_percentile is not None:
        all_err = torch.cat(frame_errs).numpy()
        tau = float(np.percentile(all_err, tau_percentile))
        print(f"tau set to P{tau_percentile} of per-object error = {tau:.2f} m")

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_fail = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for r, err in zip(records, frame_errs):
            fail_frac = float((err > tau).float().mean())
            r["fail_frac"] = fail_frac
            r["fail"] = int(fail_frac > fail_thresh)
            r["mean_err"] = float(err.mean())
            r["tau"] = tau
            n_fail += r["fail"]
            out.write(json.dumps(r) + "\n")

    print(f"Labeled {len(records)} frames -> {out_path}  "
          f"({n_fail}/{len(records)} failures at tau={tau:.2f}m, thresh={fail_thresh})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-file", default="data/output/introspection_records.jsonl")
    parser.add_argument("--checkpoint", required=True,
                        help="Trained PositionNet checkpoint (harness or bare).")
    parser.add_argument("--out-file", default="data/output/introspection_labeled.jsonl")
    parser.add_argument("--tau", type=float, default=2.0,
                        help="3D-centre error (m) above which an object is a miss.")
    parser.add_argument("--tau-percentile", type=float, default=None,
                        help="If set, derive tau from this percentile (0-100) of "
                             "the per-object error distribution instead of --tau. "
                             "Prevents a saturated (all-failure) binary label.")
    parser.add_argument("--fail-thresh", type=float, default=0.5,
                        help="fail_frac above which the frame's binary label is 1.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    label_index(args.index_file, args.checkpoint, args.out_file,
                args.tau, args.fail_thresh, device, args.tau_percentile)


if __name__ == "__main__":
    main()
