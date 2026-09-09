#!/usr/bin/env python3
"""
Advance a training checkpoint's epoch counter so `--resume` restarts at a
later epoch, skipping one that keeps failing.

This is a WORKAROUND, not a fix. It does not repair anything -- it declares an
epoch already done. Read the caveats at the bottom before relying on it.

What has to stay consistent, and how each is handled:

  epoch counter   Rewritten. runner.train() computes start_epoch = epoch + 1,
                  so to begin at epoch N this stores N - 1.

  LR scheduler    Nothing to do. No scheduler state is persisted; on resume
                  the runner fast-forwards it with
                      for _ in range((start_epoch - 1) * steps_per_epoch)
                  so it derives purely from the epoch counter and lands on the
                  correct LR for the new epoch automatically.

  GradScaler      Carried over untouched. If the checkpoint predates scaler
                  persistence, resume restarts AMP at the default scale and
                  says so; --set-scale can seed it instead.

  RNG state       Carried over untouched, so the resumed run continues the
                  saved stream rather than silently reshuffling.

  optimizer       Carried over untouched (Adam moments matter; discarding
                  them causes a destabilising first step).

  best_monitor    Carried over, so a later worse epoch cannot overwrite
                  best.pt.

  in_progress_*   Cleared. A mid-epoch snapshot's markers would otherwise
                  claim an epoch is partially done that we just declared
                  complete.

Usage:
    # inspect only
    python hipergator/bump_checkpoint_epoch.py latest.pt --to-epoch 10 --dry-run

    # write a new checkpoint (never overwrites the input)
    python hipergator/bump_checkpoint_epoch.py latest.pt --to-epoch 10 \
        --out latest_epoch10.pt
"""

import argparse
import sys

# Keys that must survive untouched for a resume to be consistent.
_PRESERVE = ("model", "optimizer", "scaler", "rng", "best_monitor", "config")


def plan_bump(ckpt_keys, current_epoch, target_epoch):
    """Pure planning step: what changes, what is preserved, what is missing.

    Separated from any torch I/O so it can be tested without a GPU or a real
    checkpoint. Returns (new_epoch_value, notes, problems).
    """
    notes, problems = [], []

    if target_epoch is None:
        problems.append("no target epoch given")
        return None, notes, problems
    if target_epoch <= current_epoch + 1:
        problems.append(
            f"--to-epoch {target_epoch} would not skip anything: this "
            f"checkpoint already resumes at epoch {current_epoch + 1}")
    new_epoch = target_epoch - 1
    skipped = new_epoch - current_epoch
    if skipped > 0:
        notes.append(f"declares epochs {current_epoch + 1}..{new_epoch} "
                     f"complete without training them ({skipped} skipped)")

    for k in _PRESERVE:
        if k in ckpt_keys:
            notes.append(f"preserved: {k}")
        else:
            notes.append(f"ABSENT (nothing to preserve): {k}")
            if k == "scaler":
                problems.append(
                    "checkpoint has no GradScaler state -- AMP will restart at "
                    "the default scale on resume (use --set-scale to seed it)")
            if k == "optimizer":
                problems.append(
                    "checkpoint has no optimizer state -- Adam moments restart "
                    "from zero, which typically destabilises the first steps")
    return new_epoch, notes, problems


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--to-epoch", type=int, required=True,
                    help="epoch the resumed run should START at")
    ap.add_argument("--out", help="output path (required unless --dry-run)")
    ap.add_argument("--set-scale", type=float, default=None,
                    help="seed the GradScaler scale (e.g. 1024) instead of "
                         "letting AMP restart at its default")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="proceed even if the checkpoint fails health checks")
    args = ap.parse_args()

    if not args.dry_run and not args.out:
        sys.exit("--out is required unless --dry-run")
    if args.out and args.out == args.checkpoint:
        sys.exit("refusing to overwrite the input checkpoint; choose another --out")

    import torch

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    current = ckpt.get("epoch")
    if current is None:
        sys.exit("checkpoint has no 'epoch' key -- is this a training checkpoint?")

    print(f"checkpoint      : {args.checkpoint}")
    print(f"stored epoch    : {current}  (resumes at epoch {current + 1})")
    print(f"target          : resume at epoch {args.to_epoch}")
    if ckpt.get("in_progress_batch"):
        print(f"note            : this is a MID-EPOCH snapshot "
              f"(epoch {ckpt.get('in_progress_epoch')}, "
              f"batch {ckpt['in_progress_batch']})")

    new_epoch, notes, problems = plan_bump(set(ckpt), current, args.to_epoch)
    print("\nplan:")
    for n in notes:
        print(f"  - {n}")

    # Health check: bumping a checkpoint that is already poisoned just moves
    # the failure, so refuse unless forced.
    print("\nhealth:")
    bad_w = [k for k, v in ckpt.get("model", {}).items()
             if torch.is_tensor(v) and v.is_floating_point()
             and not torch.isfinite(v).all()]
    print(f"  non-finite weight tensors: {len(bad_w)}"
          + (f"  e.g. {bad_w[:3]}" if bad_w else ""))
    bad_o = 0
    for st in ckpt.get("optimizer", {}).get("state", {}).values():
        for v in st.values():
            if torch.is_tensor(v) and v.is_floating_point() \
                    and not torch.isfinite(v).all():
                bad_o += 1
    print(f"  non-finite optimizer tensors: {bad_o}")
    if bad_w or bad_o:
        problems.append("checkpoint contains non-finite state; skipping an "
                        "epoch will NOT clear it")

    if problems:
        print("\nproblems:")
        for p in problems:
            print(f"  ! {p}")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return
    if problems and not args.force:
        sys.exit("\nrefusing to write; re-run with --force to override")

    ckpt["epoch"] = new_epoch
    ckpt["in_progress_epoch"] = None
    ckpt["in_progress_batch"] = None
    ckpt["epoch_bumped_from"] = current      # audit trail
    if args.set_scale is not None:
        sc = ckpt.get("scaler") or {}
        sc["scale"] = args.set_scale
        ckpt["scaler"] = sc
        print(f"\nseeded GradScaler scale = {args.set_scale}")

    tmp = args.out + ".tmp"
    torch.save(ckpt, tmp)
    import os
    os.replace(tmp, args.out)
    print(f"\nwrote {args.out}")
    print(f"  stored epoch {current} -> {new_epoch} "
          f"(resume will start at epoch {new_epoch + 1})")


if __name__ == "__main__":
    main()
