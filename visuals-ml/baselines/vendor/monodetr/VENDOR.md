# Vendored: MonoDETR

This directory contains a vendored copy of the official **MonoDETR** implementation,
used as the network/loss core for the MonoDETR baseline in `visuals-ml`.

| | |
|---|---|
| **Upstream** | https://github.com/ZrrSkywalker/MonoDETR |
| **Paper** | MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection (Zhang et al., ICCV 2023; arXiv 2203.13310) |
| **Commit** | `6994b9f512400b258c6edb75f77423beb9c126f2` |
| **Retrieved** | 2026-06-24 |
| **License** | MIT (see `LICENSE.txt` in this directory) — Copyright (c) 2024 Renrui Zhang |

## What was copied

- `lib/` — models, losses, helpers, and the KITTI dataset/eval code
- `utils/` — misc utilities
- `LICENSE.txt` — upstream MIT license (retained unmodified)

The upstream `tools/`, `configs/`, `requirements.txt`, and docs were **not** vendored;
our harness provides its own runner, configs, and Waymo data adapter.

## Local modifications

Modifications to vendored files are tracked here. Each change is also marked inline
with a `# VISUALS-MOD:` comment so it can be diffed against upstream.

Made during milestone 3 to run under our harness on modern torch (2.5):

- `lib/models/monodetr/ops/functions/ms_deform_attn_func.py` — route `.apply()` to
  the pure-PyTorch core when the compiled CUDA op is unavailable (local runs).
- `lib/models/monodetr/ops/modules/ms_deform_attn.py` — fix two torch-version
  guards that mis-parse on torch>=2 (`False < N` evaluates True), which imported
  the long-removed `_LinearWithBias` and `torch._overrides`.
- `lib/models/monodetr/monodetr.py` — (a) depth-map loss: derive grid size/device
  from the logits instead of hardcoded `[80, 24]` / `'cuda'`; (b) `loss_angles`:
  device-agnostic one-hot (was `.cuda()`); (c) `build()`: expose `group_num` via cfg
  and pass it to model + criterion (default 11 = upstream).
- `lib/models/monodetr/depthaware_transformer.py` — `build_depthaware_transformer`
  passes `group_num` from cfg so the decoder's grouping stays in sync.

Note: `group_num=1` (set in our config) makes training and inference use the same
query set, avoiding the Group-DETR train/infer mismatch for from-scratch training.
The decoder self-attention still hardcodes 50 queries-per-group (`self.group_num * 50`),
which is fine while `num_queries == 50`.

## Notes

- `lib/models/monodetr/ops/` contains a custom multi-scale deformable-attention CUDA op
  that must be compiled (`python setup.py build install`, or `make.sh`). It builds on
  Linux + CUDA (HiPerGator); it does not build cleanly on Windows. Local smoke tests
  use a pure-PyTorch fallback path instead.
- `lib/datasets/kitti/` is retained for reference but is **not** used at runtime — our
  Waymo→camera-frame adapter replaces it. It may be removed once the adapter is stable.
