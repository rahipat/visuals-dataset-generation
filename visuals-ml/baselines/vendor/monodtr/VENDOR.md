# Vendored: MonoDTR

Vendored copy of the official **MonoDTR** implementation (built on the
`visualDet3D` framework), used as the network/loss core for the MonoDTR baseline.

| | |
|---|---|
| **Upstream** | https://github.com/KuanchihHuang/MonoDTR |
| **Paper** | MonoDTR: Monocular 3D Object Detection with Depth-Aware Transformer (Huang et al., CVPR 2022; arXiv 2203.10981) |
| **Commit** | `8a40c4807b9d354d1cd12838ec094d9d70d40f72` |
| **Retrieved** | 2026-06-24 |
| **License** | MIT (see `LICENSE` in this directory) — Copyright (c) 2022 Kuan-Chih Huang |

## What was copied

- `visualDet3D/` — the full framework (backbones incl. DLA, detectors, heads, lib, ops, pipelines)
- `config/` — config module
- `LICENSE` — upstream MIT license (retained unmodified)

Upstream `launchers/`, `scripts/`, `resources/`, and docs were not vendored; our
harness provides its own runner, configs, and Waymo data adapter.

## Model entry points

- `visualDet3D/networks/detectors/monodtr_core.py` — core network (DLA-102 backbone
  + DFE/DPE/DTR modules; pure PyTorch).
- `visualDet3D/networks/detectors/monodtr_detector.py` — detector wrapper.
- `visualDet3D/networks/detectors/{dfe,dpe,dtr}.py` — depth-aware feature enhancement,
  depth positional encoding, depth-aware transformer.

## CUDA ops (compile on HiPerGator)

Two custom ops live under `visualDet3D/networks/lib/ops/`, used by the anchor-based
3D detection head (`networks/heads/detection_3d_head.py`):

- `dcn/` — deformable convolution. **Local fallback:** `torchvision.ops.deform_conv2d`
  is a drop-in for the forward path (planned VISUALS-MOD, mirroring the MonoDETR
  deformable-attn fallback).
- `iou3d/` — rotated 3D IoU + 3D NMS, used at inference (and for anchor IoU).

The pure-PyTorch core compiles cleanly; the ops build on Linux + CUDA (HiPerGator).

## Local modifications

Tracked here; each inline change is also marked `# VISUALS-MOD:`.

- _(none yet — initial import is a verbatim copy of upstream `visualDet3D/` and `config/`)_
