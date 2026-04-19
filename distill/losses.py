from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """KL distillation loss between student and teacher classification logits."""
    if student_logits.numel() == 0 or teacher_logits.numel() == 0:
        return torch.zeros((), device=student_logits.device if student_logits.numel() > 0 else teacher_logits.device)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def explanation_feature_kd_loss(student_features: torch.Tensor, teacher_features: torch.Tensor, mode: str = "cosine") -> torch.Tensor:
    """Feature-level distillation loss for explanation embeddings."""
    if student_features.numel() == 0 or teacher_features.numel() == 0:
        return torch.zeros((), device=student_features.device if student_features.numel() > 0 else teacher_features.device)
    if mode == "mse":
        return F.mse_loss(student_features, teacher_features)
    if mode != "cosine":
        raise ValueError(f"Unsupported explanation distillation mode: {mode}")
    student_norm = F.normalize(student_features, dim=-1)
    teacher_norm = F.normalize(teacher_features, dim=-1)
    cosine_sim = F.cosine_similarity(student_norm, teacher_norm, dim=-1)
    return (1.0 - cosine_sim).mean()
