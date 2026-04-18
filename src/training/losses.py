from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        targets = targets.flatten(1)
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()


@dataclass
class MultiTaskLosses:
    cls_loss: nn.Module
    loc_bce_loss: nn.Module
    loc_dice_loss: nn.Module
    exp_loss: nn.Module


class ExplanationContrastiveLoss(nn.Module):
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logits = image_features @ text_features.T
        targets = torch.arange(logits.shape[0], device=logits.device)
        loss_i = F.cross_entropy(logits, targets)
        loss_t = F.cross_entropy(logits.T, targets)
        return 0.5 * (loss_i + loss_t)


class TextFeatureEncoder(nn.Module):
    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        self.feature_dim = feature_dim

    def encode(self, texts: List[str], device: torch.device) -> torch.Tensor:
        features = torch.zeros((len(texts), self.feature_dim), dtype=torch.float32, device=device)
        for row, text in enumerate(texts):
            tokens = text.lower().split()
            if not tokens:
                tokens = ["<empty>"]
            for token in tokens:
                index = abs(hash(token)) % self.feature_dim
                features[row, index] += 1.0
        return F.normalize(features, dim=-1)


def build_multitask_losses() -> MultiTaskLosses:
    return MultiTaskLosses(
        cls_loss=nn.CrossEntropyLoss(),
        loc_bce_loss=nn.BCEWithLogitsLoss(),
        loc_dice_loss=DiceLoss(),
        exp_loss=ExplanationContrastiveLoss(),
    )
