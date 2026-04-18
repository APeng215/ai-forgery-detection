from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LocalizationHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 256) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        logits = self.decoder(x)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
