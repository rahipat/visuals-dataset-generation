"""
Model-independent train/eval runner.

Owns the epoch loop, optimizer, AMP, checkpointing, and logging. Knows nothing
about a specific baseline beyond the BaselineModel contract (core/interface.py).
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)

LOG_INTERVAL = 50
MAX_CONSECUTIVE_NONFINITE_LOSS = 5


def _maybe_cap(dataset, cfg):
    """Optionally cap a dataset to the first N samples (cfg['max_samples']) for
    fast local smoke runs. No effect when unset."""
    n = cfg.get("max_samples")
    if n and n < len(dataset):
        return Subset(dataset, list(range(n)))
    return dataset


def _build_optimizer(model, cfg):
    """Plain Adam unless cfg['weight_decay'] is set, in which case AdamW with
    decoupled weight decay (the standard choice for transformer-style models
    like MonoDETR -- see vendor/monodetr/lib/helpers/optimizer_helper.py).
    Opt-in and defaults to the old behavior so it doesn't change any config
    that doesn't ask for it."""
    weight_decay = cfg.get("weight_decay", 0.0)
    if weight_decay:
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=weight_decay)
    return torch.optim.Adam(model.parameters(), lr=cfg["lr"])


def _build_scheduler(optimizer, cfg):
    """Optional linear warmup (cfg['warmup_epochs']) composed with an optional
    step decay (cfg['lr_drop']). Mirrors the warmup+decay shape the vendored
    MonoDETR training recipe uses (vendor/monodetr/lib/helpers/trainer_helper.py
    steps a warmup scheduler for epoch<5, then a decay scheduler) -- our
    generic runner previously had neither, running a flat, undecayed LR for
    the whole run. Both keys are opt-in; returns None (flat LR, old behavior)
    if neither is set."""
    warmup_epochs = cfg.get("warmup_epochs", 0)
    lr_drop = cfg.get("lr_drop")

    schedulers, milestones = [], []
    if warmup_epochs:
        schedulers.append(torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs))
        milestones.append(warmup_epochs)
    if lr_drop is not None:
        schedulers.append(torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_drop, gamma=cfg.get("lr_drop_gamma", 0.1)))

    if not schedulers:
        return None
    if len(schedulers) == 1:
        return schedulers[0]
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=schedulers, milestones=milestones)


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
    scheduler = _build_scheduler(optimizer, cfg)
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
        best_monitor = ckpt.get("monitor", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']} (monitor={best_monitor:.4f})")

    if scheduler is not None:
        # Deterministic StepLR/LinearLR state depends only on epoch count, so
        # fast-forward by stepping once per already-completed epoch rather
        # than persisting scheduler state in the checkpoint.
        for _ in range(start_epoch - 1):
            scheduler.step()

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        train_loss = _run_train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, cfg["epochs"],
            clip_max_norm,
        )
        if scheduler is not None:
            scheduler.step()

        metrics = model.evaluate(val_loader, device)
        monitor = metrics["monitor"]
        extra = "  ".join(
            f"{k}={v:.4f}" for k, v in metrics.items()
            if k != "monitor" and isinstance(v, (int, float))
        )
        print(f"Epoch {epoch:3d}/{cfg['epochs']}  train_loss={train_loss:.4f}  "
              f"monitor={monitor:.4f}  {extra}", flush=True)

        if monitor < best_monitor:
            best_monitor = monitor
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "monitor": monitor, "metrics": metrics, "config": cfg},
                checkpoint_dir / "best.pt",
            )
            print(f"  --> checkpoint saved (monitor={monitor:.4f})", flush=True)


def _run_train_epoch(model, loader, optimizer, scaler, device, epoch, total_epochs,
                     clip_max_norm=None):
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
            # compute on a gradient that can only be garbage.
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

        bs = logs.get("batch_size", 1)
        total_loss += loss.item() * bs
        n_samples += bs

        if (batch_idx + 1) % LOG_INTERVAL == 0 or (batch_idx + 1) == n_batches:
            extra = "  ".join(f"{k}={v:.4f}" for k, v in logs.items() if k != "batch_size")
            print(f"  [train] epoch {epoch}/{total_epochs}  "
                  f"batch {batch_idx+1}/{n_batches}  loss={loss.item():.4f}  {extra}",
                  flush=True)

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
