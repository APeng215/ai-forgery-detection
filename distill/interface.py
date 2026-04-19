from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

import torch
from torch import nn

from .cache import OfflineTeacherCache
from .losses import classification_kd_loss, explanation_feature_kd_loss


def init_ema_teacher(student_model: nn.Module, device: torch.device) -> nn.Module:
    teacher_model = deepcopy(student_model).to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)
    return teacher_model


def update_ema_teacher(teacher_model: nn.Module, student_model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=(1.0 - decay))
        for teacher_buffer, student_buffer in zip(teacher_model.buffers(), student_model.buffers()):
            teacher_buffer.copy_(student_buffer)


def compute_online_distill_loss(
    outputs: Dict[str, torch.Tensor],
    teacher_outputs: Dict[str, torch.Tensor] | None,
    batch: Dict,
    device: torch.device,
    config: Dict,
) -> tuple[torch.Tensor, Dict[str, float]]:
    zero = torch.tensor(0.0, device=device)
    if teacher_outputs is None:
        return zero, {}

    temperature = float(config.get("temperature", 4.0))
    total_weight = float(config.get("total_weight", 0.5))
    cls_weight = float(config.get("cls_weight", 1.0))
    exp_weight = float(config.get("exp_weight", 0.5))
    exp_mode = str(config.get("exp_mode", "cosine"))

    total_distill = torch.tensor(0.0, device=device)
    scalars: Dict[str, float] = {}

    if batch["has_cls"].any():
        cls_mask = batch["has_cls"]
        cls_loss = classification_kd_loss(
            outputs["logits"][cls_mask],
            teacher_outputs["logits"][cls_mask].detach(),
            temperature=temperature,
        )
        total_distill = total_distill + cls_weight * cls_loss
        scalars["distill_cls_loss"] = float(cls_loss.item())

    if batch["has_exp"].any():
        exp_mask = batch["has_exp"]
        exp_loss = explanation_feature_kd_loss(
            outputs["explanation_features"][exp_mask],
            teacher_outputs["explanation_features"][exp_mask].detach(),
            mode=exp_mode,
        )
        total_distill = total_distill + exp_weight * exp_loss
        scalars["distill_exp_loss"] = float(exp_loss.item())

    if total_distill.item() == 0.0:
        return zero, scalars

    scaled_total = total_weight * total_distill
    scalars["distill_loss"] = float(scaled_total.item())
    return scaled_total, scalars


class DistillProvider:
    """Compute optional distillation losses from offline teacher cache."""

    def __init__(self, config: Dict, device: torch.device) -> None:
        self.cfg = config or {}
        self.device = device
        self.enabled = bool(self.cfg.get("enabled", False))
        self.stage_b_only = bool(self.cfg.get("stage_b_only", True))
        self.temperature = float(self.cfg.get("temperature", 4.0))
        self.cls_weight = float(self.cfg.get("cls_weight", 1.0))
        self.exp_weight = float(self.cfg.get("exp_weight", 0.5))
        self.total_weight = float(self.cfg.get("total_weight", 0.5))
        self.exp_mode = str(self.cfg.get("exp_mode", "cosine"))
        self.strict = bool(self.cfg.get("strict", False))

        self.cache: Optional[OfflineTeacherCache] = None
        if self.enabled:
            mode = self.cfg.get("mode", "offline_cache")
            if mode == "offline_cache":
                cache_path = self.cfg.get("cache_path", "")
                if not cache_path:
                    raise ValueError("distill.cache_path must be provided when distill.enabled is true and mode=offline_cache")
                self.cache = OfflineTeacherCache(cache_path=cache_path, strict=self.strict)
            elif mode == "ema_online":
                self.cache = None
            else:
                raise ValueError(f"Unsupported distill mode: {mode}")

    def is_active(self, stage_label: str) -> bool:
        if not self.enabled:
            return False
        if self.stage_b_only and stage_label != "B":
            return False
        return True

    def compute(self, outputs: Dict[str, torch.Tensor], batch: Dict, stage_label: str) -> tuple[torch.Tensor, Dict[str, float]]:
        zero = torch.tensor(0.0, device=self.device)
        if not self.is_active(stage_label) or self.cache is None:
            return zero, {}

        image_paths = batch.get("image_path", [])
        if not image_paths:
            return zero, {"distill_coverage": 0.0}

        entries = self.cache.get_many(image_paths)
        available_indices = [idx for idx, entry in enumerate(entries) if entry is not None]
        coverage = len(available_indices) / max(len(image_paths), 1)

        if not available_indices:
            return zero, {"distill_coverage": 0.0}

        total_distill = torch.tensor(0.0, device=self.device)
        scalars: Dict[str, float] = {"distill_coverage": coverage}

        # Classification distillation.
        cls_indices = []
        cls_teacher_logits = []
        has_cls = batch.get("has_cls")
        for idx in available_indices:
            entry = entries[idx]
            if entry is None or "logits" not in entry:
                continue
            if has_cls is not None and not bool(has_cls[idx].item()):
                continue
            cls_indices.append(idx)
            cls_teacher_logits.append(entry["logits"])
        if cls_indices:
            cls_index_tensor = torch.tensor(cls_indices, device=self.device, dtype=torch.long)
            teacher_logits = torch.stack(cls_teacher_logits).to(self.device)
            student_logits = outputs["logits"].index_select(0, cls_index_tensor)
            distill_cls = classification_kd_loss(student_logits, teacher_logits, temperature=self.temperature)
            total_distill = total_distill + self.cls_weight * distill_cls
            scalars["distill_cls_loss"] = float(distill_cls.item())

        # Explanation feature distillation.
        exp_indices = []
        exp_teacher_features = []
        has_exp = batch.get("has_exp")
        for idx in available_indices:
            entry = entries[idx]
            if entry is None or "explanation_features" not in entry:
                continue
            if has_exp is not None and not bool(has_exp[idx].item()):
                continue
            exp_indices.append(idx)
            exp_teacher_features.append(entry["explanation_features"])
        if exp_indices:
            exp_index_tensor = torch.tensor(exp_indices, device=self.device, dtype=torch.long)
            teacher_features = torch.stack(exp_teacher_features).to(self.device)
            student_features = outputs["explanation_features"].index_select(0, exp_index_tensor)
            distill_exp = explanation_feature_kd_loss(student_features, teacher_features, mode=self.exp_mode)
            total_distill = total_distill + self.exp_weight * distill_exp
            scalars["distill_exp_loss"] = float(distill_exp.item())

        if total_distill.item() == 0.0:
            return zero, scalars

        scaled_total = self.total_weight * total_distill
        scalars["distill_loss"] = float(scaled_total.item())
        return scaled_total, scalars
