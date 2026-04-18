from __future__ import annotations

import torch
from torch import nn

from .backbone import ResNet50Backbone
from .heads.classification_head import ClassificationHead
from .heads.localization_head import LocalizationHead
from .heads.explanation_head import ExplanationHead


class MultiTaskForgeryModel(nn.Module):
    def __init__(self, pretrained_backbone: bool = True) -> None:
        super().__init__()
        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)
        self.classification_head = ClassificationHead(self.backbone.out_channels)
        self.localization_head = LocalizationHead(self.backbone.out_channels)
        self.explanation_head = ExplanationHead(self.backbone.out_channels)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(image)
        high = features["high"]
        logits = self.classification_head(high)
        mask_logits = self.localization_head(high, output_size=(image.shape[-2], image.shape[-1]))
        explanation_features = self.explanation_head(high)
        return {
            "logits": logits,
            "mask_logits": mask_logits,
            "explanation_features": explanation_features,
        }
