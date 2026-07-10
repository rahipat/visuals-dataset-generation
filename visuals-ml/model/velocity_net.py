"""
Siamese two-stream ResNet18 → velocity prediction.

crop_t   (3, 224, 224) → ResNet18(shared) → AvgPool → 512-d ┐
crop_t+1 (3, 224, 224) → ResNet18(shared) → AvgPool → 512-d ┤
coords   (15,)  ─────────────────────────────────────────────┘
                                        concat → 1039-d
                                  MLP: 1039 → 256 → 64 → 2

coords layout (15-d):
    cx_n_t, cy_n_t, sw_n_t, sh_n_t       — box at frame T (normalised)
    cx_n_t1, cy_n_t1, sw_n_t1, sh_n_t1  — box at frame T+1
    ego_linear_x/y/z                     — ego linear velocity (m/s)
    ego_angular_x/y/z                    — ego angular velocity (rad/s)
    delta_t                              — time between frames (s)

output (2,): [vx, vy] in vehicle frame (m/s)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def _build_encoder() -> nn.Sequential:
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    return nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
        backbone.layer4,
    )


class VelocityNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Shared encoder: same weights for both frames (siamese)
        self.encoder = _build_encoder()
        self.pool    = nn.AdaptiveAvgPool2d(1)

        # 512 + 512 + 15 = 1039
        self.head = nn.Sequential(
            nn.Linear(512 + 512 + 15, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        crop_t:  torch.Tensor,   # (B, 3, 224, 224)
        crop_t1: torch.Tensor,   # (B, 3, 224, 224)
        coords:  torch.Tensor,   # (B, 15)
    ) -> torch.Tensor:           # (B, 2)
        feat_t  = self.pool(self.encoder(crop_t)).flatten(1)   # (B, 512)
        feat_t1 = self.pool(self.encoder(crop_t1)).flatten(1)  # (B, 512)
        x = torch.cat([feat_t, feat_t1, coords], dim=1)        # (B, 1039)
        return self.head(x)                                     # (B, 2)
