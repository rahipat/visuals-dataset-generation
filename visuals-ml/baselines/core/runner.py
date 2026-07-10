"""
Model-independent train/eval runner.

Owns the epoch loop, optimizer, AMP, checkpointing, and logging. Knows nothing
about a specific baseline beyond the BaselineModel contract (core/interface.py).
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

LOG_INTERVAL = 50


def _maybe_cap(dataset, cfg):
    """Optionally cap a dataset to the first N samples (cfg['max_samples']) for
    fast local smoke runs. No effect when unset."""
    n = cfg.get("max_samples")
    if n and n < len(dataset):
        return Subset(dataset, list(range(n)))
    return dataset


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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
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
        start_epoch = ckpt["epoch"] + 1
        best_monitor = ckpt.get("monitor", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']} (monitor={best_monitor:.4f})")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        train_loss = _run_train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, cfg["epochs"],
            clip_max_norm,
        )

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

    for batch_idx, batch in enumerate(loader):
        with torch.autocast(device_type=device.type, enabled=use_cuda):
            loss, logs = model.training_step(batch, device)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if clip_max_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
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
