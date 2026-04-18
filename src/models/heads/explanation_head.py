from __future__ import annotations

from typing import List

import torch
from torch import nn


class ExplanationHead(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int = 256) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Linear(in_channels, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)
        return self.projector(pooled)

    def predict(self, features: torch.Tensor, candidates: List[str], candidate_features: torch.Tensor) -> List[str]:
        if len(candidates) == 0:
            return [""] * features.shape[0]
        normalized_features = torch.nn.functional.normalize(features, dim=-1)
        normalized_candidates = torch.nn.functional.normalize(candidate_features, dim=-1)
        scores = normalized_features @ normalized_candidates.T
        indices = scores.argmax(dim=1).tolist()
        return [candidates[i] for i in indices]
