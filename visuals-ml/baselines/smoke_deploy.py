"""
Deploy smoke test — one command, machine-readable verdict.

PURPOSE. Decide whether the full HiPerGator training jobs are worth scheduling,
without a human reading logs. Runs every configured baseline briefly on a small
subset, applies explicit pass/fail gates, and writes a JSON report plus a short
text summary. Exit code is 0 only if every gate passes.

It is a DEPLOY test, not a training run. It answers "will the full job run, and
is the setup sane?" -- not "is the model good". Nothing here trains to
convergence and no number produced here belongs in a paper.

WHAT IT CHECKS, in order (later stages are skipped when an earlier one fails,
because their results would be meaningless):

  A. Environment   python/torch/CUDA present, GPU visible and usable, the
                   optional deps each model needs importable.
  B. Data          index file exists, is non-empty, paths inside it resolve to
                   readable images, expected weather variants present, enough
                   distinct segments to form a leak-free split.
  C. Build         each model instantiates and reports its parameter count.
  D. Step          one forward+backward actually runs; loss is finite.
  E. Overfit       a handful of batches are fit repeatedly; the loss must DROP
                   by a minimum factor. This is the real wiring test -- it
                   catches broken targets, detached graphs and bad decodes that
                   a single step cannot.
  F. Eval          the eval pass runs and returns a finite `monitor`.
  G. Throughput    measured samples/sec, extrapolated to a full-run estimate so
                   the wall-clock ask can be sanity-checked before submitting.

Usage (from visuals-ml/):
    python -m baselines.smoke_deploy --config baselines/configs/smoke_deploy.yaml
    python -m baselines.smoke_deploy --models box3d --report out.json

Read the verdict from the JSON, not the log:
    {"verdict": "PASS"|"FAIL", "models": {...}, "gates": {...}}
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import torch

import baselines.models  # noqa: F401  (registers every baseline)
from baselines.core.registry import available_models, build_model
from baselines.core.utils import load_config

# --- gates ------------------------------------------------------------------
# Deliberately loose. They are wiring/deploy gates, not quality bars: a model
# that clears them is worth a real run, not necessarily any good.
DEFAULT_GATES = {
    # Loss must fall to at most this fraction of its starting value while
    # overfitting a few fixed batches. Anything that cannot fit data it has seen
    # dozens of times is broken, not undertrained.
    "overfit_loss_ratio": 0.6,
    "min_segments": 2,       # below this a leak-free split is impossible
    "min_images_checked": 5,
    "min_throughput": 0.5,   # samples/sec; below this a full run is infeasible
}


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Stage:
    """One check: records ok/skip/fail plus detail, and never raises."""

    def __init__(self, results: OrderedDict, name: str):
        self.results, self.name = results, name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def ok(self, **detail):
        self.results[self.name] = {"status": "ok", **detail}

    def warn(self, reason, **detail):
        """Ran correctly, but something is worth a human's attention. Does not
        fail the verdict."""
        self.results[self.name] = {"status": "warn", "reason": str(reason), **detail}

    def fail(self, reason, **detail):
        self.results[self.name] = {"status": "fail", "reason": str(reason), **detail}

    def skip(self, reason):
        self.results[self.name] = {"status": "skip", "reason": str(reason)}

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self.results[self.name] = {
                "status": "fail",
                "reason": f"{exc_type.__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
        entry = self.results.setdefault(self.name, {"status": "fail",
                                                    "reason": "stage did not report"})
        entry["seconds"] = round(time.time() - self.t0, 2)
        return True   # never propagate; the report is the product


# --- A. environment ---------------------------------------------------------

def check_environment(gates) -> dict:
    out = OrderedDict()
    with Stage(out, "python_torch") as s:
        s.ok(python=sys.version.split()[0], torch=torch.__version__,
             platform=platform.platform(), cuda_build=torch.version.cuda)

    with Stage(out, "gpu") as s:
        if not torch.cuda.is_available():
            s.fail("torch.cuda.is_available() is False — the job asked for a GPU "
                   "but none is visible. Check #SBATCH --gpus and --partition, "
                   "and that the container ran with `singularity exec --nv`.")
        else:
            i = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(i)
            # Prove the GPU actually computes, not just that it enumerates.
            a = torch.randn(256, 256, device="cuda")
            val = float((a @ a).sum())
            s.ok(name=props.name, count=torch.cuda.device_count(),
                 total_memory_gb=round(props.total_memory / 1024**3, 1),
                 capability=f"{props.major}.{props.minor}",
                 matmul_finite=bool(val == val))

    with Stage(out, "optional_deps") as s:
        found = {}
        for mod in ("numpy", "yaml", "PIL", "torchvision", "sklearn", "cv2"):
            try:
                __import__(mod)
                found[mod] = True
            except Exception:
                found[mod] = False
        # sklearn is only needed by the introspection SVM stage, cv2 only by its
        # optical-flow stream; neither blocks the perception baselines.
        s.ok(**found)
    return out


# --- B. data ----------------------------------------------------------------

def check_data(index_file: str, gates) -> dict:
    out = OrderedDict()
    path = Path(index_file)

    with Stage(out, "index_file") as s:
        if not path.exists():
            s.fail(f"index not found: {path} — run the matching "
                   "baselines.data.build_*_index first")
        else:
            n = sum(1 for line in open(path, encoding="utf-8") if line.strip())
            if n == 0:
                s.fail(f"index {path} is empty")
            else:
                s.ok(path=str(path), records=n,
                     size_mb=round(path.stat().st_size / 1024**2, 1))

    if out["index_file"]["status"] != "ok":
        for k in ("segments", "images_readable", "weather_coverage"):
            out[k] = {"status": "skip", "reason": "no usable index"}
        return out

    records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    with Stage(out, "segments") as s:
        segs = {r.get("segment") for r in records if r.get("segment")}
        if not segs:
            s.fail("no `segment` field in the index — rebuild it; the "
                   "leak-free split cannot be formed without segment ids")
        elif len(segs) < gates["min_segments"]:
            s.fail(f"only {len(segs)} segment(s); need >= {gates['min_segments']} "
                   "for a segment-level train/val split")
        else:
            s.ok(n_segments=len(segs))

    with Stage(out, "images_readable") as s:
        from PIL import Image
        n_check = min(gates["min_images_checked"] * 4, len(records))
        step = max(1, len(records) // max(n_check, 1))
        checked, bad = 0, []
        for r in records[::step][:n_check]:
            p = r.get("image_path") or r.get("image_t")
            if not p:
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
                checked += 1
            except Exception as e:
                bad.append({"path": str(p), "error": str(e)})
        if checked < gates["min_images_checked"]:
            s.fail(f"only {checked} image(s) readable out of {n_check} sampled — "
                   "image paths in the index are absolute; if the index was "
                   "built elsewhere they will not resolve here. Rebuild it on "
                   "this filesystem.", bad_samples=bad[:5])
        else:
            s.ok(checked=checked, unreadable=len(bad), bad_samples=bad[:5])

    with Stage(out, "weather_coverage") as s:
        seen = {}
        for r in records:
            w = r.get("weather", "unknown")
            seen[w] = seen.get(w, 0) + 1
        s.ok(variants=len(seen), counts=dict(sorted(seen.items())))
    return out


# --- C-G. per model ---------------------------------------------------------

def _first_batches(model, cfg, device, n_batches):
    from torch.utils.data import DataLoader, Subset
    train_set, _val = model.build_datasets(cfg)
    n = min(len(train_set), cfg["batch_size"] * n_batches)
    subset = Subset(train_set, list(range(n)))
    loader = DataLoader(subset, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=0, collate_fn=model.collate_fn)
    return list(loader), train_set, _val


def check_model(name: str, cfg: dict, device, gates, n_batches, overfit_iters) -> dict:
    out = OrderedDict()
    model = None

    with Stage(out, "build") as s:
        model = build_model({**cfg, "model": name})
        model.to(device)
        s.ok(params_m=round(sum(p.numel() for p in model.parameters()) / 1e6, 2))
    if out["build"]["status"] != "ok":
        for k in ("data_batches", "train_step", "overfit", "evaluate", "throughput"):
            out[k] = {"status": "skip", "reason": "model failed to build"}
        return out

    batches = None
    with Stage(out, "data_batches") as s:
        batches, train_set, val_set = _first_batches(model, cfg, device, n_batches)
        if not batches:
            s.fail("dataset produced zero batches — the index has no usable "
                   "samples after filtering")
        else:
            s.ok(n_batches=len(batches), batch_size=cfg["batch_size"],
                 train_samples=len(train_set), val_samples=len(val_set))
    if out["data_batches"]["status"] != "ok":
        for k in ("train_step", "overfit", "evaluate", "throughput"):
            out[k] = {"status": "skip", "reason": "no batches"}
        return out

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    with Stage(out, "train_step") as s:
        model.train()
        loss, logs = model.training_step(batches[0], device)
        if not torch.isfinite(loss):
            s.fail(f"loss is not finite: {loss.item()}")
        else:
            optimizer.zero_grad()
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            total_norm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads)))
            optimizer.step()
            if not (total_norm == total_norm):        # NaN check
                s.fail("gradient norm is NaN after one backward pass")
            elif total_norm == 0.0:
                s.fail("gradient norm is exactly 0 — nothing is connected to the "
                       "loss; check that training_step uses the model's output")
            else:
                s.ok(loss=round(float(loss), 4), grad_norm=round(total_norm, 4),
                     logs={k: round(float(v), 4) for k, v in logs.items()})
    if out["train_step"]["status"] != "ok":
        for k in ("overfit", "evaluate", "throughput"):
            out[k] = {"status": "skip", "reason": "single train step failed"}
        return out

    with Stage(out, "overfit") as s:
        model.train()
        first = last = None
        history = []
        for it in range(overfit_iters):
            epoch_loss, n = 0.0, 0
            for b in batches:
                loss, logs = model.training_step(b, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bs = logs.get("batch_size", 1)
                epoch_loss += float(loss) * bs
                n += bs
            avg = epoch_loss / max(n, 1)
            history.append(round(avg, 4))
            first = avg if first is None else first
            last = avg
        ratio = last / first if first else float("inf")
        detail = {"first_loss": round(first, 4), "last_loss": round(last, 4),
                  "ratio": round(ratio, 4), "iters": overfit_iters,
                  "history": history}
        if not (ratio == ratio):
            s.fail("loss became NaN while overfitting", **detail)
        elif ratio > gates["overfit_loss_ratio"]:
            s.fail(f"loss fell to {ratio:.2f}x of its start; gate requires "
                   f"<= {gates['overfit_loss_ratio']}. The model cannot fit "
                   "batches it has seen repeatedly, which points at the targets "
                   "or the decode, not at needing more epochs.", **detail)
        else:
            s.ok(**detail)

    with Stage(out, "evaluate") as s:
        from torch.utils.data import DataLoader, Subset
        _train, val_set = model.build_datasets(cfg)
        n = min(len(val_set), cfg["batch_size"] * n_batches)
        loader = DataLoader(Subset(val_set, list(range(n))),
                            batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=0, collate_fn=model.collate_fn)
        metrics = model.evaluate(loader, device)
        mon = metrics.get("monitor")
        recall = metrics.get("recall")
        if mon is None:
            s.fail("evaluate() returned no 'monitor' key (required by the runner)")
        elif not isinstance(mon, (int, float)) or mon != mon:
            s.fail(f"monitor is not a usable float: {mon!r}. The runner "
                   "checkpoints on this value; NaN means no checkpoint is ever "
                   "saved.", metrics=_jsonable(metrics))
        elif mon == float("inf"):
            # For a detector this is the CORRECT score when nothing is detected,
            # and a model given ~36 optimiser steps detects nothing. Distinguish
            # "untrained" from "broken": broken would have failed the overfit
            # stage above. Warn so a real run is still gated on seeing recall
            # climb off zero in the first epochs.
            if recall == 0 or recall is None:
                s.warn("monitor is +inf because the model detected nothing at "
                       "this score threshold. Expected for an untrained "
                       "detector (the overfit stage already proved the loss "
                       "descends). In the full run, recall must leave 0 within "
                       "the first few epochs — if it does not, the decode or "
                       "score_thresh is wrong, not the training length.",
                       recall=recall, metrics=_jsonable(metrics))
            else:
                s.fail(f"monitor is +inf despite recall={recall}; the metric is "
                       "inconsistent.", metrics=_jsonable(metrics))
        else:
            s.ok(monitor=round(float(mon), 4), metrics=_jsonable(metrics))

    with Stage(out, "throughput") as s:
        model.train()
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        n_samples = 0
        for b in batches:
            loss, logs = model.training_step(b, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_samples += logs.get("batch_size", cfg["batch_size"])
        torch.cuda.synchronize() if device.type == "cuda" else None
        dt = time.time() - t0
        rate = n_samples / dt if dt > 0 else 0.0
        if rate < gates["min_throughput"]:
            s.fail(f"{rate:.2f} samples/s is below the {gates['min_throughput']} "
                   "gate; a full run would not finish in any reasonable walltime")
        else:
            s.ok(samples_per_sec=round(rate, 2), measured_samples=n_samples,
                 seconds=round(dt, 2))
    return out


def _jsonable(obj):
    """Make metrics JSON-safe (numpy scalars, inf, nested dicts)."""
    import math
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            obj = obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return round(obj, 6)
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


def estimate_full_run(rate, samples, epochs):
    """Wall-clock estimate for a full run at the measured rate."""
    if not rate or rate <= 0:
        return None
    secs = samples * epochs / rate
    return {"samples": samples, "epochs": epochs,
            "hours": round(secs / 3600, 1),
            "note": "measured on the smoke subset with num_workers=0; a real "
                    "run with dataloader workers is usually faster, so treat "
                    "this as an upper bound."}


def collect_status(tree) -> list:
    """Flatten to a list of (path, status) for the verdict."""
    flat = []

    def walk(node, prefix):
        for k, v in node.items():
            if isinstance(v, dict) and "status" in v:
                flat.append((f"{prefix}{k}", v["status"]))
            elif isinstance(v, dict):
                walk(v, f"{prefix}{k}.")
    walk(tree, "")
    return flat


def merge_reports(paths, out_path: Path) -> int:
    """Combine per-model reports (one per array task) into one verdict file.

    A missing file is itself a failure — it means the array task crashed before
    writing, which must not be read as 'no problems found'.
    """
    merged = OrderedDict()
    merged["generated_at_utc"] = _now()
    merged["merged_from"] = [str(p) for p in paths]
    merged["environment"] = None
    merged["data"] = OrderedDict()
    merged["models"] = OrderedDict()
    missing = []

    for p in paths:
        path = Path(p)
        if not path.exists():
            missing.append(str(path))
            continue
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        if merged["environment"] is None:
            merged["environment"] = r.get("environment")
            merged["gates"] = r.get("gates")
            merged["registered_models"] = r.get("registered_models")
        merged["data"].update(r.get("data") or {})
        merged["models"].update(r.get("models") or {})

    statuses = collect_status({
        "environment": merged.get("environment") or {},
        "data": merged["data"],
        "models": merged["models"],
    })
    n_fail = sum(1 for _p, s in statuses if s == "fail")
    n_skip = sum(1 for _p, s in statuses if s == "skip")
    n_warn = sum(1 for _p, s in statuses if s == "warn")
    merged["summary"] = {
        "checks": len(statuses),
        "passed": sum(1 for _p, s in statuses if s == "ok"),
        "failed": n_fail,
        "skipped": n_skip,
        "warned": n_warn,
        "missing_reports": missing,
        "failures": [p for p, s in statuses if s == "fail"],
        "warnings": [p for p, s in statuses if s == "warn"],
    }
    merged["verdict"] = (
        "PASS" if n_fail == 0 and n_skip == 0 and not missing else "FAIL"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print("=" * 60)
    print(f"MERGED VERDICT: {merged['verdict']}   "
          f"({merged['summary']['passed']} passed, {n_fail} failed, "
          f"{n_skip} skipped, {n_warn} warned)")
    for p in missing:
        print(f"  MISSING REPORT: {p}  (array task crashed before writing)")
    for p in merged["summary"]["failures"]:
        print(f"  FAIL: {p}")
    for p in merged["summary"]["warnings"]:
        print(f"  WARN: {p}")
    print(f"Report: {out_path}")
    print("=" * 60)
    return 0 if merged["verdict"] == "PASS" else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baselines/configs/smoke_deploy.yaml",
                    help="Smoke config: per-model overrides + gates.")
    ap.add_argument("--models", default=None,
                    help="Comma-separated subset of the configured models.")
    ap.add_argument("--report", default=None, help="JSON report path.")
    ap.add_argument("--batches", type=int, default=None)
    ap.add_argument("--overfit-iters", type=int, default=None)
    ap.add_argument("--index-suffix", default=None,
                    help="Insert '_<suffix>' before .jsonl in every index_file, "
                         "so smoke and full indexes can coexist "
                         "(det_records.jsonl -> det_records_smoke.jsonl).")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="Merge per-model reports into one verdict file and "
                         "exit. Used by the aggregation job.")
    args = ap.parse_args()

    if args.merge:
        return merge_reports(args.merge, Path(args.report or "smoke_report.json"))

    cfg = load_config(args.config)
    gates = {**DEFAULT_GATES, **(cfg.get("gates") or {})}
    n_batches = args.batches or cfg.get("batches", 3)
    overfit_iters = args.overfit_iters or cfg.get("overfit_iters", 12)
    report_path = Path(args.report or cfg.get("report", "smoke_report.json"))

    model_cfgs = cfg.get("models") or {}
    if args.models:
        want = [m.strip() for m in args.models.split(",")]
        missing = [m for m in want if m not in model_cfgs]
        if missing:
            ap.error(f"not configured: {missing}. Configured: {sorted(model_cfgs)}")
        model_cfgs = {m: model_cfgs[m] for m in want}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== deploy smoke test ===  {_now()}")
    print(f"device={device}  models={sorted(model_cfgs)}  "
          f"batches={n_batches}  overfit_iters={overfit_iters}\n")

    report = OrderedDict()
    report["generated_at_utc"] = _now()
    report["host"] = platform.node()
    report["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    report["device"] = str(device)
    report["registered_models"] = available_models()
    report["gates"] = gates

    report["environment"] = check_environment(gates)
    env_ok = report["environment"]["gpu"]["status"] == "ok"

    report["data"] = OrderedDict()
    report["models"] = OrderedDict()

    for name, overrides in model_cfgs.items():
        merged = {**(cfg.get("common") or {}), **(overrides or {})}
        if args.index_suffix and merged.get("index_file"):
            p = Path(merged["index_file"])
            merged["index_file"] = str(
                p.with_name(f"{p.stem}_{args.index_suffix}{p.suffix}"))
        print(f"--- {name} ---", flush=True)

        idx = merged.get("index_file")
        if idx and idx not in report["data"]:
            report["data"][idx] = check_data(idx, gates)
        data_ok = all(
            v["status"] in ("ok", "skip")
            for v in report["data"].get(idx, {}).values()
        ) and report["data"].get(idx, {}).get("index_file", {}).get("status") == "ok"

        if not env_ok:
            report["models"][name] = {
                "_": {"status": "skip", "reason": "no usable GPU"}}
        elif idx and not data_ok:
            report["models"][name] = {
                "_": {"status": "skip", "reason": f"data checks failed for {idx}"}}
        else:
            report["models"][name] = check_model(
                name, merged, device, gates, n_batches, overfit_iters)
            tp = report["models"][name].get("throughput", {})
            if tp.get("status") == "ok" and merged.get("full_run_samples"):
                report["models"][name]["full_run_estimate"] = estimate_full_run(
                    tp["samples_per_sec"], merged["full_run_samples"],
                    merged.get("full_run_epochs", 30))

        for stage, res in report["models"][name].items():
            if isinstance(res, dict) and "status" in res:
                mark = {"ok": "PASS", "fail": "FAIL",
                        "skip": "SKIP", "warn": "WARN"}[res["status"]]
                line = f"  [{mark}] {stage}"
                if res["status"] != "ok":
                    line += f" — {res.get('reason', '')}"
                print(line, flush=True)
        print(flush=True)

    statuses = collect_status(
        {"environment": report["environment"], "data": report["data"],
         "models": report["models"]})
    n_fail = sum(1 for _p, s in statuses if s == "fail")
    n_skip = sum(1 for _p, s in statuses if s == "skip")
    n_warn = sum(1 for _p, s in statuses if s == "warn")
    report["summary"] = {
        "checks": len(statuses),
        "passed": sum(1 for _p, s in statuses if s == "ok"),
        "failed": n_fail,
        "skipped": n_skip,
        "warned": n_warn,
        "failures": [p for p, s in statuses if s == "fail"],
        "warnings": [p for p, s in statuses if s == "warn"],
    }
    # Skips count against the verdict: a check that did not run is not a check
    # that passed. Warnings do not — they are for a human, not a gate.
    report["verdict"] = "PASS" if n_fail == 0 and n_skip == 0 else "FAIL"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print(f"VERDICT: {report['verdict']}   "
          f"({report['summary']['passed']} passed, {n_fail} failed, "
          f"{n_skip} skipped, {n_warn} warned)")
    if report["summary"]["failures"]:
        print("Failed checks:")
        for p in report["summary"]["failures"]:
            print(f"  - {p}")
    if report["summary"]["warnings"]:
        print("Warnings (not blocking):")
        for p in report["summary"]["warnings"]:
            print(f"  - {p}")
    print(f"Report: {report_path}")
    print("=" * 60)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
