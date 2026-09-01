"""
Model-independent train/eval runner.

Owns the epoch loop, optimizer, AMP, checkpointing, and logging. Knows nothing
about a specific baseline beyond the BaselineModel contract (core/interface.py).
"""

import logging
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)

LOG_INTERVAL = 50
MEMORY_LOG_INTERVAL = 1000
MAX_CONSECUTIVE_NONFINITE_LOSS = 5


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

    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_cuda):
            loss, logs = model.training_step(batch, device)

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
