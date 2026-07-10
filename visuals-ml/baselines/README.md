# baselines

A model-agnostic harness for running paper baselines on the visuals (weather-altered
Waymo) dataset. Define a model once against a small interface, then train / evaluate it
locally and on HiPerGator through the same runner, with per-weather-variant metrics.

## Layout

```
baselines/
  core/
    interface.py   # BaselineModel — the contract every baseline implements
    registry.py    # @register_model("name") + build_model(cfg)
    runner.py      # model-independent train/eval loop, AMP, checkpointing
    boxes.py       # shared Box3D (camera optical frame) for detection baselines
    utils.py       # config loading, deterministic train/val split
  models/
    position_net.py  # PositionNet (per-object 3D-center regressor) — reference baseline
    monodetr/        # MonoDETR (per-image detector) — added in milestone 2
  vendor/
    monodetr/        # vendored upstream MonoDETR (MIT) — see vendor/monodetr/VENDOR.md
  configs/
  train.py / eval.py # thin entrypoints; pick the model from config `model:` key
```

## Adding a baseline

Implement `BaselineModel` (see `core/interface.py`) and register it:

```python
@register_model("mymodel")
class MyBaseline(BaselineModel):
    @staticmethod
    def build_datasets(cfg): ...      # -> (train_ds, val_ds)
    def collate_fn(self, batch): ...  # override for variable-length detection batches
    def training_step(self, batch, device): ...  # -> (loss, log_dict)
    def evaluate(self, loader, device): ...       # -> metrics dict incl. float "monitor"
```

Then add `from baselines.models import mymodel` to `models/__init__.py` and a config with
`model: mymodel`.

**Contract notes**
- `evaluate()` must return a dict with a float `"monitor"` key where **lower is better**
  (for higher-is-better metrics like AP, return the negative). The runner checkpoints
  whenever `monitor` improves.
- `training_step` runs inside the runner's autocast context — return the loss, do **not**
  call `backward()`.

## Usage

Run from the `visuals-ml/` directory.

```bash
# train (model selected by the config's `model:` key)
python -m baselines.train --config configs/positionnet.yaml

# evaluate a checkpoint (defaults to <checkpoint_dir>/best.pt)
python -m baselines.eval --config configs/positionnet.yaml
```

Useful config keys: `max_samples` caps the dataset for fast local smoke runs;
`num_workers: 0` avoids Windows dataloader issues. See `configs/positionnet.yaml`.
