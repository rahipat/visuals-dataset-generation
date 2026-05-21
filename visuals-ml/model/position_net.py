"""
Two-stream ResNet18 encoder + coord-vector fusion → 3D position prediction.

Full image  (3, 640, 960)  → ResNet18(full) → AvgPool → 512-d
Box crop    (3, 224, 224)  → ResNet18(crop) → AvgPool → 512-d
Coords      (8,)           ─────────────────────────────────┐
                                                       concat → 1032-d
                                                 MLP: 1032 → 256 → 128 → 3
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


class PositionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_full = _build_encoder()
        self.encoder_crop = _build_encoder()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(512 + 512 + 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        image: torch.Tensor,
        crop: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        feat_full = self.pool(self.encoder_full(image)).flatten(1)   # (B, 512)
        feat_crop = self.pool(self.encoder_crop(crop)).flatten(1)    # (B, 512)
        x = torch.cat([feat_full, feat_crop, coords], dim=1)         # (B, 1032)
        return self.head(x)                                          # (B, 3)
