"""
Model-independent train/eval runner.

Owns the epoch loop, optimizer, AMP, checkpointing, and logging. Knows nothing
about a specific baseline beyond the BaselineModel contract (core/interface.py).
"""

import logging
import math
import random
import time
from collections import deque
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)

LOG_INTERVAL = 50
MEMORY_LOG_INTERVAL = 1000
MAX_CONSECUTIVE_NONFINITE_LOSS = 5
# Every batch is recorded here, not just logged ones: when a run goes
# non-finite the interesting window is the handful of batches immediately
# before it, and LOG_INTERVAL sampling is far too coarse to show whether a
# quantity ramped or jumped.
RECENT_HISTORY = 30


def _maybe_cap(dataset, cfg):
    """Optionally cap a dataset to the first N samples (cfg['max_samples']) for
    fast local smoke runs. No effect when unset."""
    n = cfg.get("max_samples")
    if n and n < len(dataset):
        return Subset(dataset, list(range(n)))
    return dataset


def _build_optimizer(model, cfg):
    """Adam, or AdamW when cfg['weight_decay'] is set. cfg['lr_backbone'] puts
    backbone parameters in their own lower-LR group -- the DETR-family
    convention (backbone is pretrained and needs far gentler updates than the
    randomly-initialised transformer/heads). MonoDETR trains its backbone
    (train_backbone: True in the adapter) but the harness previously drove
    every parameter at one LR. All keys opt-in; defaults reproduce the old
    plain-Adam behavior for configs that don't set them."""
    weight_decay = cfg.get("weight_decay", 0.0)
    lr = cfg["lr"]
    lr_backbone = cfg.get("lr_backbone")

    params = model.parameters()
    if lr_backbone is not None:
        backbone, rest = [], []
        for name, p in model.named_parameters():
            if p.requires_grad:
                (backbone if "backbone" in name else rest).append(p)
        if not backbone:
            # Name-matching is a convention, not a contract -- don't silently
            # train everything at the wrong LR if a model names things
            # differently.
            logger.warning(
                "lr_backbone=%g set but no parameter name contains 'backbone'; "
                "using a single LR group.", lr_backbone)
        else:
            params = [{"params": rest, "lr": lr},
                      {"params": backbone, "lr": lr_backbone}]
            print(f"Optimizer param groups: {len(rest)} @ lr={lr:g}, "
                  f"{len(backbone)} backbone @ lr={lr_backbone:g}")

    if weight_decay:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr)


def _build_scheduler(optimizer, cfg, steps_per_epoch, total_epochs=None):
    """Per-ITERATION linear warmup (cfg['warmup_iters']) followed by decay.

    cfg['lr_schedule'] picks the decay shape:
      'step'   (default) -- 10x cliff every cfg['lr_drop'] epochs.
      'cosine'           -- smooth decay from full LR down to
                            cfg['lr_min_factor'] (default 0.01) across the run.

    Cosine exists because a step cliff only helps if training survives long
    enough to reach it: MonoDETR diverged at epoch 9 with the drop set at
    epoch 20, so the LR it actually trained under was constant the whole time.
    A schedule that decays continuously lowers the LR as the model's state
    changes, instead of betting everything on one milestone.

    Deliberately iteration-granular: this dataset runs ~29k batches per epoch,
    so an epoch-granular warmup is a coarse staircase that jumps most of the
    way to full LR after a single epoch -- it does not do the thing warmup
    exists to do, which is keep the first few hundred/thousand steps small
    while Adam's second-moment estimates are still noisy. LambdaLR scales each
    param group off its own initial_lr, so a separate backbone LR group is
    warmed and decayed proportionally. Returns None (flat LR) when unset."""
    warmup_iters = cfg.get("warmup_iters", 0)
    lr_drop = cfg.get("lr_drop")
    gamma = cfg.get("lr_drop_gamma", 0.1)
    start_factor = cfg.get("warmup_start_factor", 0.01)
    shape = cfg.get("lr_schedule", "step")
    min_factor = cfg.get("lr_min_factor", 0.01)

    total_steps = (total_epochs or 0) * steps_per_epoch
    if shape == "cosine" and total_steps <= warmup_iters:
        logger.warning("lr_schedule='cosine' needs total steps > warmup_iters; "
                       "falling back to a flat post-warmup LR.")
        shape = "step"
        lr_drop = None

    if not warmup_iters and lr_drop is None and shape != "cosine":
        return None

    def factor(step):
        if warmup_iters and step < warmup_iters:
            return start_factor + (1.0 - start_factor) * (step / warmup_iters)
        if shape == "cosine":
            progress = ((step - warmup_iters)
                        / max(1, total_steps - warmup_iters))
            progress = min(1.0, max(0.0, progress))
            return min_factor + (1.0 - min_factor) * 0.5 * (
                1.0 + math.cos(math.pi * progress))
        if lr_drop and steps_per_epoch:
            return gamma ** ((step // steps_per_epoch) // lr_drop)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _code_version():
    """Git SHA of the code actually running, best-effort.

    When an expected diagnostic is missing from a log, the first question is
    always 'which commit produced this log?' -- and answering it has required
    cross-referencing the repo by hand. A submitted job can easily predate the
    commit that added the instrumentation being looked for, and nothing in the
    output said so. Now it does.

    Reads .git directly rather than shelling out to git: the training container
    has no git binary, so the subprocess version reported
    'unknown (FileNotFoundError)' on exactly the runs where knowing the commit
    mattered most.
    """
    try:
        here = Path(__file__).resolve().parent
        git_dir = None
        for d in [here, *here.parents]:
            cand = d / ".git"
            if cand.exists():
                git_dir = cand
                break
        if git_dir is None:
            return "unknown (no .git found)"

        # A worktree's .git is a file containing 'gitdir: <path>'.
        if git_dir.is_file():
            txt = git_dir.read_text(encoding="utf-8").strip()
            if not txt.startswith("gitdir:"):
                return "unknown (unreadable .git file)"
            git_dir = Path(txt.split(":", 1)[1].strip())

        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head[:9]  # detached HEAD

        # In a linked worktree, HEAD is worktree-local but refs live in the
        # main repository's git dir, named by 'commondir'.
        common = git_dir
        commondir = git_dir / "commondir"
        if commondir.exists():
            common = (git_dir / commondir.read_text(encoding="utf-8").strip()).resolve()

        ref = head.split(":", 1)[1].strip()
        for base in (git_dir, common):
            loose = base / ref
            if loose.exists():
                return loose.read_text(encoding="utf-8").strip()[:9]

        packed = common / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.startswith(("#", "^")):
                    continue
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0][:9]
        return f"unknown (ref {ref} unresolved)"
    except Exception as e:
        return f"unknown ({type(e).__name__})"


# Diagnostics this version of the runner can emit on a failure. Printed at
# startup so a log states its own capabilities rather than leaving their
# absence ambiguous.
DIAGNOSTICS = (
    "per-batch-history", "weight-fingerprint", "forward-module-hooks",
    "checkpoint-audit", "scaler+rng-restore",
)


def _locate_forward_nan(model, batch, device, use_cuda, limit=6):
    """Re-run ONE batch with forward hooks on every submodule to find where a
    non-finite value first enters the forward pass.

    Only called after a batch has already failed, so the cost (a device sync
    per module) is paid once, never during healthy training. Hooks fire in
    execution order, so the first module reporting a non-finite OUTPUT while
    its INPUTS were still finite is the origin -- everything downstream of it
    is just propagation. A module with non-finite inputs is listed too, but
    marked as a victim rather than the cause."""
    found = []
    handles = []

    def _tensors(x):
        if torch.is_tensor(x):
            return [x]
        if isinstance(x, (list, tuple)):
            return [t for t in x if torch.is_tensor(t)]
        if isinstance(x, dict):
            return [t for t in x.values() if torch.is_tensor(t)]
        return []

    def _nonfinite(ts):
        return any(t.is_floating_point() and not torch.isfinite(t).all() for t in ts)

    def make_hook(name):
        def hook(mod, inp, out):
            if len(found) >= limit:
                return
            out_bad = _nonfinite(_tensors(out))
            if not out_bad:
                return
            in_bad = _nonfinite(_tensors(inp))
            found.append({
                "module": name or "<root>",
                "type": type(mod).__name__,
                "inputs_finite": not in_bad,
            })
        return hook

    for name, mod in model.named_modules():
        handles.append(mod.register_forward_hook(make_hook(name)))
    try:
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=use_cuda):
                model.training_step(batch, device)
    except Exception as e:
        logger.warning("forward-NaN localisation re-run failed: %s", e)
    finally:
        for h in handles:
            h.remove()
    return found


def _param_fingerprint(model):
    """Cheap GPU-side fingerprint of all weights. Kept as a tensor so it costs
    no host sync per batch; only compared (and synced) when something fails.

    Its job is to answer one question that theory alone cannot: did the last
    optimizer step ACTUALLY apply? GradScaler is supposed to skip the step
    whenever it finds inf/NaN gradients, so a halved scale should imply
    unchanged weights. If the fingerprint moves across a batch where the scale
    was backed off, that assumption is false and the search moves to the
    optimizer/scaler interaction rather than the data."""
    total = None
    for p in model.parameters():
        if p.is_floating_point():
            s = p.detach().sum()
            total = s if total is None else total + s
    return total


def _rng_state():
    """Capture every RNG that affects a training run. Without these, a resume
    replays the epoch with a DIFFERENT shuffle order and different dropout
    masks than the run being resumed -- so 'resume and see if it happens
    again' silently tests a different data sequence each time."""
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as _np
        state["numpy"] = _np.random.get_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return False
    try:
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu")
                            else state["torch"])
        if "numpy" in state:
            import numpy as _np
            _np.random.set_state(state["numpy"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
        return True
    except Exception as e:
        logger.warning("Could not restore RNG state (%s); shuffle order will "
                       "differ from the interrupted run.", e)
        return False


def _nonfinite_optimizer_state(optimizer, limit=8):
    """Non-finite entries in the optimizer's own state (Adam's exp_avg /
    exp_avg_sq).

    Worth checking separately from the weights: a checkpoint can hold perfectly
    finite parameters alongside a poisoned second-moment buffer, and nothing
    shows up until the first step that actually applies -- at which point
    exp_avg/sqrt(exp_avg_sq) can be inf/inf = NaN and the weights die in one
    update, long after the checkpoint 'looked' clean."""
    bad = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p)
            if not st:
                continue
            for key, val in st.items():
                if torch.is_tensor(val) and val.is_floating_point() \
                        and not torch.isfinite(val).all():
                    bad.append(f"{key}{tuple(val.shape)}")
                    if len(bad) >= limit:
                        return bad + ["..."]
    return bad


def _nonfinite_params(model, limit=8):
    """Names of parameters containing NaN/Inf.

    This is the decisive test when a loss goes non-finite. If the weights are
    still finite, the forward pass turned healthy weights into NaN for this
    particular batch -- look at the batch/inputs. If the weights are already
    non-finite, the corruption happened during an EARLIER optimizer step and
    the non-finite loss is just the first place it became visible; the real
    event is upstream and the batch here is innocent."""
    bad = []
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            bad.append(name)
            if len(bad) >= limit:
                bad.append("...")
                break
    return bad


def _dump_recent(recent, epoch, n_batches):
    """Replay the per-batch history leading into a failure, at full
    resolution, so a slow ramp is distinguishable from a single-step jump."""
    if not recent:
        return
    print(f"\n  ==== per-batch history into the failure "
          f"(epoch {epoch}, last {len(recent)} batches) ====", flush=True)
    for batch, loss_s, scale, stats in recent:
        print(f"    batch {batch:>6}/{n_batches}  loss={loss_s:>12}  "
              f"amp_scale={scale:<12.6g} {stats}", flush=True)
    print("  ==== end history ====\n", flush=True)


def _atomic_save(state, path):
    """Write via a temp file + rename so a crash mid-write can't leave a
    truncated checkpoint where a resumable one used to be."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _memory_report():
    """One-line host/GPU memory snapshot. Worker RSS is the number that matters
    here: forked DataLoader workers are separate processes, so the parent's own
    RSS hides most of a dataset-side leak."""
    parts = []
    try:
        import os
        import psutil
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss
        kids = [c.memory_info().rss for c in proc.children(recursive=True)]
        parts.append(f"rss={rss/1e9:.2f}GB")
        if kids:
            parts.append(f"workers={sum(kids)/1e9:.2f}GB(n={len(kids)},"
                         f"max={max(kids)/1e9:.2f}GB)")
        parts.append(f"host_total={(rss + sum(kids))/1e9:.2f}GB")
    except ImportError:
        try:
            import resource
            import sys as _sys
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss is bytes on macOS, kilobytes on Linux.
            scale = 1e9 if _sys.platform == "darwin" else 1e6
            parts.append(f"peak_rss={peak/scale:.2f}GB (install psutil for "
                         "per-worker RSS)")
        except Exception:
            return "memory: unavailable"
    except Exception as e:  # psutil present but query failed
        return f"memory: unavailable ({e})"

    if torch.cuda.is_available():
        parts.append(f"cuda_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
                     f"cuda_peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    return "  ".join(parts)


def _make_loader(dataset, model, cfg, *, shuffle, device):
    pin = device.type == "cuda"
    workers = cfg.get("num_workers", 0)
    persistent = workers > 0
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin,
        collate_fn=model.collate_fn,
        persistent_workers=persistent,
        prefetch_factor=4 if persistent else None,
    )


def train(model, cfg, device, resume=None):
    train_set, val_set = model.build_datasets(cfg)
    train_set, val_set = _maybe_cap(train_set, cfg), _maybe_cap(val_set, cfg)
    print(f"Train: {len(train_set)}  Val: {len(val_set)}")

    train_loader = _make_loader(train_set, model, cfg, shuffle=True, device=device)
    val_loader = _make_loader(val_set, model, cfg, shuffle=False, device=device)

    model.to(device)
    optimizer = _build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader)
    scheduler = _build_scheduler(optimizer, cfg, steps_per_epoch, cfg["epochs"])
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    clip_max_norm = cfg.get("clip_max_norm")  # None disables clipping

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_monitor = float("inf")
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            print("  (no optimizer state in checkpoint -- Adam moments restart from scratch)")

        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
            print(f"  restored GradScaler (scale={scaler.get_scale():g})")
        else:
            print(f"  WARNING: checkpoint has no GradScaler state -- AMP restarts "
                  f"at the default scale ({scaler.get_scale():g}), not whatever "
                  f"the interrupted run had settled to. Expect skipped steps "
                  f"and halving until it re-converges.")

        if _restore_rng_state(ckpt.get("rng")):
            print("  restored RNG state (shuffle order continues the original run)")
        else:
            print("  WARNING: checkpoint has no RNG state -- this epoch replays "
                  "with a DIFFERENT shuffle order than the run being resumed, "
                  "so a data-dependent failure will land at a different batch.")

        # Audit what was actually loaded. A checkpoint can carry finite weights
        # alongside poisoned optimizer moments; that stays invisible until the
        # first applied step, which is exactly the 'healthy batch 1, dead batch
        # 2' shape.
        bad_p = _nonfinite_params(model)
        bad_o = _nonfinite_optimizer_state(optimizer)
        if bad_p:
            logger.error("RESUMED CHECKPOINT HAS NON-FINITE WEIGHTS: %s -- this "
                         "checkpoint is already dead; resume from an earlier one.",
                         ", ".join(bad_p))
        if bad_o:
            logger.error("RESUMED CHECKPOINT HAS NON-FINITE OPTIMIZER STATE: %s -- "
                         "weights may look fine, but the first applied step will "
                         "propagate NaN into them.", ", ".join(bad_o))
        if not bad_p and not bad_o:
            print("  checkpoint health: weights and optimizer state all finite")

        start_epoch = ckpt["epoch"] + 1
        # Prefer the running best over this checkpoint's own monitor: latest.pt
        # is whatever ran last, not necessarily the best, so keying off its
        # monitor would let a worse epoch overwrite best.pt after a resume.
        best_monitor = ckpt.get("best_monitor")
        if best_monitor is None:
            best_monitor = ckpt.get("monitor") or float("inf")
        print(f"Resumed from epoch {ckpt['epoch']} (best_monitor={best_monitor:.4f})")
        if ckpt.get("in_progress_batch"):
            print(f"  (checkpoint was a mid-epoch snapshot from epoch "
                  f"{ckpt['in_progress_epoch']} batch {ckpt['in_progress_batch']}; "
                  f"epoch {ckpt['in_progress_epoch']} restarts from its beginning)")

    if scheduler is not None and start_epoch > 1:
        # The LR factor is a pure function of the global step count, so
        # fast-forward instead of persisting scheduler state in the checkpoint.
        for _ in range((start_epoch - 1) * steps_per_epoch):
            scheduler.step()
        print(f"Scheduler fast-forwarded to step "
              f"{(start_epoch - 1) * steps_per_epoch}  "
              f"lr={optimizer.param_groups[0]['lr']:.3g}")

    print(f"Runner code version: {_code_version()}", flush=True)
    print(f"Failure diagnostics active: {', '.join(DIAGNOSTICS)}", flush=True)
    print(f"Startup memory: {_memory_report()}", flush=True)

    # Epochs here are enormous (PositionNet: ~170k batches, many hours), so
    # per-epoch checkpointing is still coarse. Snapshot mid-epoch too when
    # asked. A mid-epoch snapshot records the LAST FULLY COMPLETED epoch, so
    # resuming replays the interrupted epoch from its start rather than
    # silently skipping the batches it never got to.
    ckpt_every = cfg.get("checkpoint_every_n_batches")

    def _snapshot(epoch, *, completed_epoch, train_loss=None, metrics=None,
                  monitor=None, batch=None):
        _atomic_save({
            "epoch": completed_epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            # Without the scaler state a resume restarts AMP at the default
            # init_scale (65536) regardless of what the run had settled to,
            # so the first few batches back can overflow and skip steps.
            "scaler": scaler.state_dict(),
            "rng": _rng_state(),
            "best_monitor": best_monitor,   # so resume keeps the running best
            "monitor": monitor,
            "metrics": metrics,
            "train_loss": train_loss,
            "in_progress_epoch": epoch if batch is not None else None,
            "in_progress_batch": batch,
            "config": cfg,
        }, checkpoint_dir / "latest.pt")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        on_batch_ckpt = None
        if ckpt_every:
            def on_batch_ckpt(batch_idx, _epoch=epoch):
                _snapshot(_epoch, completed_epoch=_epoch - 1, batch=batch_idx)
                print(f"  --> latest.pt snapshot (epoch {_epoch}, "
                      f"batch {batch_idx}; resume replays this epoch)",
                      flush=True)

        train_loss = _run_train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, cfg["epochs"],
            clip_max_norm, scheduler, on_batch_ckpt, ckpt_every,
        )

        # Save BEFORE validating. Validation is a full pass over the val split
        # (~42k batches for PositionNet) and emits no output, so a crash, hang,
        # or preemption in there used to discard the entire epoch of training
        # that had just finished.
        _snapshot(epoch, completed_epoch=epoch, train_loss=train_loss)
        print(f"Epoch {epoch:3d}/{cfg['epochs']}  train_loss={train_loss:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.3g}", flush=True)
        print(f"  --> latest.pt saved (epoch {epoch}, pre-validation)", flush=True)
        print(f"  memory: {_memory_report()}", flush=True)

        # Announce the validation pass: it is long and silent, and its silence
        # has already been mistaken for a hang.
        print(f"  validating ({len(val_loader)} batches)...", flush=True)
        t0 = time.time()
        metrics = model.evaluate(val_loader, device)
        val_secs = time.time() - t0

        monitor = metrics["monitor"]
        extra = "  ".join(
            f"{k}={v:.4f}" for k, v in metrics.items()
            if k != "monitor" and isinstance(v, (int, float))
        )
        print(f"  validated in {val_secs:.0f}s  monitor={monitor:.4f}  {extra}",
              flush=True)

        if monitor < best_monitor:
            best_monitor = monitor
            _atomic_save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "rng": _rng_state(),
                "best_monitor": best_monitor,
                "monitor": monitor, "metrics": metrics,
                "train_loss": train_loss, "config": cfg,
            }, checkpoint_dir / "best.pt")
            print(f"  --> best.pt saved (monitor={monitor:.4f})", flush=True)

        # Refresh latest.pt with the metrics now that validation has run.
        _snapshot(epoch, completed_epoch=epoch, train_loss=train_loss,
                  metrics=metrics, monitor=monitor)


def _run_train_epoch(model, loader, optimizer, scaler, device, epoch, total_epochs,
                     clip_max_norm=None, scheduler=None,
                     on_batch_checkpoint=None, checkpoint_every=None):
    model.train()
    total_loss = 0.0
    n_samples = 0
    n_batches = len(loader)
    use_cuda = device.type == "cuda"
    consecutive_nonfinite = 0
    recent = deque(maxlen=RECENT_HISTORY)
    dumped_history = False
    prev_fp = None          # weight fingerprint at the END of the previous batch
    prev_scale = None

    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        fp_before = _param_fingerprint(model)
        scale_before = scaler.get_scale() if scaler.is_enabled() else float("nan")
        with torch.autocast(device_type=device.type, enabled=use_cuda):
            loss, logs = model.training_step(batch, device)

        # Record EVERY batch (the logs values are already-synced floats, so
        # this costs nothing beyond the deque) -- the window right before a
        # failure is exactly what LOG_INTERVAL sampling throws away.
        stats = "  ".join(
            f"{k}={v:.4g}" for k, v in logs.items()
            if k != "batch_size" and isinstance(v, (int, float))
        )
        recent.append((batch_idx + 1, f"{loss.item():.6g}",
                       scaler.get_scale() if scaler.is_enabled() else float("nan"),
                       stats))

        if not torch.isfinite(loss):
            consecutive_nonfinite += 1
            extra = "  ".join(
                f"{k}={v:.4f}" for k, v in logs.items()
                if k != "batch_size" and isinstance(v, (int, float))
            )
            logger.error(
                "epoch %d/%d batch %d/%d: non-finite loss (%s)  %s  "
                "[%d/%d consecutive]",
                epoch, total_epochs, batch_idx + 1, n_batches, loss.item(),
                extra, consecutive_nonfinite, MAX_CONSECUTIVE_NONFINITE_LOSS,
            )
            # Non-numeric diagnostics (the model's own autopsy of where the
            # NaN first appears) are dropped by the float formatter above, so
            # surface them explicitly.
            autopsy = "  ".join(f"{k}={v}" for k, v in logs.items()
                                if isinstance(v, str))
            if autopsy:
                logger.error("  first-NaN localisation: %s", autopsy)
            # Dump the deep diagnostics on the first non-finite batch AND
            # again on the batch that actually aborts the run. Previously this
            # was first-only: a single transient non-finite batch that later
            # recovered would consume the one dump, and the fatal failure
            # thousands of batches later printed only the summary -- which is
            # precisely the "hooks say active but never trace" symptom.
            fatal = consecutive_nonfinite >= MAX_CONSECUTIVE_NONFINITE_LOSS
            if not dumped_history or fatal:
                dumped_history = True
                logger.error("  deep diagnostics (%s):",
                             "FATAL batch" if fatal else "first non-finite batch")
                _dump_recent(recent, epoch, n_batches)

                # Did the PREVIOUS batch's optimizer step actually apply?
                # GradScaler backing the scale off is supposed to mean "step
                # skipped, weights untouched". If the weights moved anyway,
                # that assumption is wrong and this is not a data problem.
                if prev_fp is not None:
                    delta = (fp_before - prev_fp).abs().item()
                    backed_off = (prev_scale is not None
                                  and scale_before < prev_scale)
                    logger.error(
                        "weight fingerprint changed by %.6g since the previous "
                        "batch; AMP scale %s (%g -> %g). Expected: change==0 "
                        "when the scale backs off (step skipped). %s",
                        delta,
                        "BACKED OFF" if backed_off else "steady",
                        prev_scale if prev_scale is not None else float("nan"),
                        scale_before,
                        "CONTRADICTION: weights moved on a skipped step -- "
                        "investigate the optimizer/GradScaler interaction, not "
                        "the data." if (backed_off and delta > 0) else
                        "Consistent with GradScaler skipping the step."
                        if backed_off else
                        "Step applied normally (scale steady).",
                    )
                # Are the weights themselves already corrupt? This decides
                # whether the culprit is this batch or an earlier update.
                # If the forward itself died, re-run this exact batch with
                # per-module hooks to find where the NaN enters. Only reached
                # once, on the first failure.
                origin = _locate_forward_nan(model, batch, device, use_cuda)
                if origin:
                    logger.error("  forward-NaN origin (execution order):")
                    for i, m in enumerate(origin):
                        role = ("ORIGIN (inputs were finite)" if m["inputs_finite"]
                                else "downstream victim (inputs already bad)")
                        logger.error("    %d. %s [%s] -- %s",
                                     i + 1, m["module"], m["type"], role)
                else:
                    logger.error("  no module produced a non-finite output on "
                                 "re-run -- the failure is not reproducible from "
                                 "this batch's inputs alone (points at state: "
                                 "weights, optimizer, or AMP), or it lives in a "
                                 "functional op outside any nn.Module.")

                bad = _nonfinite_params(model)
                if bad:
                    logger.error(
                        "MODEL WEIGHTS are already non-finite (%d shown): %s -- "
                        "corruption predates this batch; the forward pass had no "
                        "chance of producing a finite loss. Look for the earlier "
                        "optimizer step that admitted it, not at this batch.",
                        len(bad), ", ".join(bad))
                else:
                    logger.error(
                        "Model weights are all FINITE -- this batch's forward pass "
                        "produced NaN/Inf from healthy weights. The trigger is in "
                        "this batch's inputs or activations, not accumulated "
                        "weight corruption.")
            if consecutive_nonfinite >= MAX_CONSECUTIVE_NONFINITE_LOSS:
                raise RuntimeError(
                    f"Training diverged: loss was non-finite for "
                    f"{consecutive_nonfinite} consecutive batches (epoch {epoch}, "
                    f"batch {batch_idx + 1}/{n_batches}). Stopping now instead of "
                    "burning further compute on a dead run. The last saved "
                    "checkpoint predates this (checkpoints only save on strict "
                    "monitor improvement, which a NaN run can't produce), so "
                    "resuming from it is safe -- but check the LR schedule / "
                    "weight decay before restarting, or this will likely recur."
                )
            # Known-bad batch: skip backward/step entirely rather than waste
            # compute on a gradient that can only be garbage. The LR schedule
            # still advances, so the warmup/decay curve stays tied to the
            # global step count rather than drifting on skipped batches.
            if scheduler is not None:
                scheduler.step()
            continue

        consecutive_nonfinite = 0
        scaler.scale(loss).backward()

        if clip_max_norm:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            if not torch.isfinite(grad_norm):
                logger.warning(
                    "epoch %d/%d batch %d/%d: non-finite grad norm (%s) despite a "
                    "finite loss -- forward was clean, backward diverged. "
                    "GradScaler should skip this optimizer step.",
                    epoch, total_epochs, batch_idx + 1, n_batches, grad_norm.item(),
                )
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        # Carry this batch's pre-step fingerprint/scale forward, so the next
        # batch can tell whether the step it just ran actually moved the
        # weights (see the autopsy in the non-finite branch above).
        prev_fp, prev_scale = fp_before, scale_before

        bs = logs.get("batch_size", 1)
        total_loss += loss.item() * bs
        n_samples += bs

        if (batch_idx + 1) % LOG_INTERVAL == 0 or (batch_idx + 1) == n_batches:
            extra = "  ".join(f"{k}={v:.4f}" for k, v in logs.items() if k != "batch_size")
            print(f"  [train] epoch {epoch}/{total_epochs}  "
                  f"batch {batch_idx+1}/{n_batches}  loss={loss.item():.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.3g}  {extra}",
                  flush=True)

        # Periodic memory trace: a steadily climbing host_total/workers figure
        # across an epoch is the signature of a dataset-side leak, as opposed
        # to a one-off spike from a large batch.
        if (batch_idx + 1) % MEMORY_LOG_INTERVAL == 0:
            print(f"  [mem] epoch {epoch}/{total_epochs}  "
                  f"batch {batch_idx+1}/{n_batches}  {_memory_report()}",
                  flush=True)

        if (on_batch_checkpoint is not None and checkpoint_every
                and (batch_idx + 1) % checkpoint_every == 0):
            on_batch_checkpoint(batch_idx + 1)

    return total_loss / max(n_samples, 1)


def evaluate(model, cfg, device, checkpoint):
    _, val_set = model.build_datasets(cfg)
    val_set = _maybe_cap(val_set, cfg)
    val_loader = _make_loader(val_set, model, cfg, shuffle=False, device=device)

    model.to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (monitor={ckpt.get('monitor', float('nan')):.4f})")

    metrics = model.evaluate(val_loader, device)
    return metrics
