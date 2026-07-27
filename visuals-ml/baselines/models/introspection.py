"""
Introspective Perception (Paper 1, Daftry et al. 2016) under the harness.

Faithful reimplementation (no official code exists). The predictor is the paper's
multi-step pipeline:

  1. A two-stream AlexNet-style ConvNet — a spatial stream over the RGB frame and
     a temporal stream over stacked optical flow — whose fc7 outputs (4096 each)
     are concatenated and fused by an fc8 layer into a shared deep feature.
  2. The CNN is trained with a softmax loss against the binary failure label
     (introspection_label_prepass.py). This runs in the harness's Adam runner.
  3. After CNN training, fc8 features feed a linear SVM that regresses the
     continuous failure fraction -> the final failure score in [0, 1]. The SVM
     stage lives in baselines/introspection_svm.py (it needs the train split),
     using extract_fc8() here.

The BaselineModel.evaluate() contract reports CNN-head metrics (failure-prediction
AUROC + per-weather) so the runner has a valid 'monitor' to checkpoint on; the
paper's final Error-vs-Failure-Rate curve is produced by the SVM stage.

Deviations from the paper (documented): AlexNet input is 224 (torchvision
pretrained) rather than 227; the temporal stream's conv1 is channel-inflated from
ImageNet weights instead of UCF-101/HMDB-51 flow pretraining, which we do not
have. See baselines/README.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet, AlexNet_Weights

from baselines.core.interface import BaselineModel
from baselines.core.registry import register_model
from baselines.core.utils import split_dataset
from baselines.data.introspection_dataset import (
    IntrospectionDataset, load_labeled_records,
)

FC7_DIM = 4096


class _AlexStream(nn.Module):
    """AlexNet conv1-5 + fc6 + fc7 -> 4096-d feature. `in_ch` != 3 rebuilds conv1
    by channel-inflating the pretrained ImageNet filters (temporal stream)."""

    def __init__(self, in_ch=3):
        super().__init__()
        net = alexnet(weights=AlexNet_Weights.DEFAULT)
        self.features = net.features
        self.avgpool = net.avgpool
        # classifier[:6] == Dropout, fc6, ReLU, Dropout, fc7, ReLU  -> 4096
        self.fc = nn.Sequential(*list(net.classifier.children())[:6])
        if in_ch != 3:
            self._inflate_conv1(in_ch)

    def _inflate_conv1(self, in_ch):
        old = self.features[0]  # Conv2d(3, 64, 11, stride=4, padding=2)
        new = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding)
        with torch.no_grad():
            mean_w = old.weight.mean(dim=1, keepdim=True)      # (64,1,11,11)
            new.weight.copy_(mean_w.repeat(1, in_ch, 1, 1) * (3.0 / in_ch))
            new.bias.copy_(old.bias)
        self.features[0] = new

    def forward(self, x):
        x = self.avgpool(self.features(x))
        return self.fc(torch.flatten(x, 1))                    # (B, 4096)


class IntrospectionNet(nn.Module):
    """Two-stream ConvNet + fc8 fusion + softmax failure head."""

    def __init__(self, flow_ch, feat_dim=512):
        super().__init__()
        self.spatial = _AlexStream(in_ch=3)
        self.temporal = _AlexStream(in_ch=flow_ch)
        self.fc8 = nn.Sequential(nn.Linear(2 * FC7_DIM, feat_dim), nn.ReLU(inplace=True))
        self.head = nn.Linear(feat_dim, 2)                     # softmax: reliable / fail

    def extract_fc8(self, spatial, flow):
        f = torch.cat([self.spatial(spatial), self.temporal(flow)], dim=1)
        return self.fc8(f)                                     # (B, feat_dim)

    def forward(self, spatial, flow):
        return self.head(self.extract_fc8(spatial, flow))      # (B, 2)


def _auroc(labels, scores):
    """Mann-Whitney U AUROC (numpy-only). labels in {0,1}."""
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_sum = ranks[labels].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@register_model("introspection")
class IntrospectionBaseline(BaselineModel):
    def __init__(self, cfg: dict):
        super().__init__()
        self.resolution = cfg.get("resolution", 224)
        self.flow_stack = cfg.get("flow_stack", 5)
        self.flow_cache = cfg.get("flow_cache")
        self.net = IntrospectionNet(
            flow_ch=2 * self.flow_stack, feat_dim=cfg.get("feat_dim", 512))
        self.criterion = nn.CrossEntropyLoss()

    # ---- data ----------------------------------------------------------------
    def build_datasets(self, cfg: dict):
        records = load_labeled_records(cfg["index_file"])
        print(f"Introspection frames: {len(records)}")
        train_split, val_split = split_dataset(records, cfg["val_split"], cfg["seed"])
        train_recs = [records[i] for i in train_split.indices]
        val_recs = [records[i] for i in val_split.indices]
        make = lambda recs: IntrospectionDataset(
            recs, resolution=self.resolution, flow_cache=self.flow_cache)
        return make(train_recs), make(val_recs)

    def collate_fn(self, batch):
        spatial = torch.stack([b[0] for b in batch])
        flow = torch.stack([b[1] for b in batch])
        fail_frac = torch.stack([b[2] for b in batch])
        fail = torch.stack([b[3] for b in batch])
        mean_err = torch.stack([b[4] for b in batch])
        weather = [b[5] for b in batch]
        return spatial, flow, fail_frac, fail, mean_err, weather

    # ---- train ---------------------------------------------------------------
    def training_step(self, batch, device):
        spatial, flow, _fail_frac, fail, _mean_err, _weather = batch
        spatial, flow, fail = spatial.to(device), flow.to(device), fail.to(device)
        logits = self.net(spatial, flow)
        loss = self.criterion(logits, fail)
        return loss, {"ce": loss.item(), "batch_size": len(fail)}

    # ---- eval (CNN head; the SVM stage produces the paper's EFR curve) --------
    @torch.no_grad()
    def evaluate(self, loader, device) -> dict:
        self.eval()
        scores, fails = [], []
        per_w_score = defaultdict(list)
        per_w_err = defaultdict(list)
        for spatial, flow, fail_frac, fail, _mean_err, weather in loader:
            spatial, flow = spatial.to(device), flow.to(device)
            p_fail = F.softmax(self.net(spatial, flow), dim=1)[:, 1].cpu()
            scores.append(p_fail)
            fails.append(fail)
            for i, w in enumerate(weather):
                per_w_score[w].append(float(p_fail[i]))
                per_w_err[w].append(float(fail_frac[i]))

        scores = torch.cat(scores).numpy()
        fails = torch.cat(fails).numpy()
        auroc = _auroc(fails, scores)
        metrics = {
            # monitor: lower is better -> negative AUROC
            "monitor": float(-auroc) if auroc == auroc else 0.0,
            "auroc": float(auroc),
            "fail_rate": float(fails.mean()),
            "n": int(len(fails)),
        }
        metrics["per_weather"] = {
            w: {
                "n": len(per_w_score[w]),
                "mean_pred_fail": float(sum(per_w_score[w]) / len(per_w_score[w])),
                "mean_true_fail_frac": float(sum(per_w_err[w]) / len(per_w_err[w])),
            }
            for w in sorted(per_w_score)
        }
        return metrics

    # ---- used by the SVM stage ----------------------------------------------
    @torch.no_grad()
    def extract_fc8(self, loader, device):
        """Return (features (N, feat_dim), fail_frac (N,), mean_err (N,),
        weather list) for the linear-SVM stage."""
        self.eval()
        feats, fracs, errs, weathers = [], [], [], []
        for spatial, flow, fail_frac, _fail, mean_err, weather in loader:
            spatial, flow = spatial.to(device), flow.to(device)
            feats.append(self.net.extract_fc8(spatial, flow).cpu())
            fracs.append(fail_frac)
            errs.append(mean_err)
            weathers.extend(weather)
        return torch.cat(feats), torch.cat(fracs), torch.cat(errs), weathers
