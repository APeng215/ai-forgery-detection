from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

import torch


def binary_average_precision(targets: torch.Tensor, scores: torch.Tensor) -> float:
    if targets.numel() == 0:
        return 0.0
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_targets = targets[sorted_indices]
    positives = sorted_targets.sum().item()
    if positives == 0:
        return 0.0
    cumulative = torch.cumsum(sorted_targets, dim=0)
    precision = cumulative / (torch.arange(sorted_targets.numel(), device=sorted_targets.device) + 1)
    ap = (precision * sorted_targets).sum() / positives
    return float(ap.item())


def compute_bleu_1(predictions: List[str], references: List[str]) -> float:
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        if not pred_tokens:
            scores.append(0.0)
            continue
        ref_counts = defaultdict(int)
        for token in ref_tokens:
            ref_counts[token] += 1
        hits = 0
        used = defaultdict(int)
        for token in pred_tokens:
            if used[token] < ref_counts[token]:
                hits += 1
                used[token] += 1
        scores.append(hits / len(pred_tokens))
    return float(sum(scores) / max(len(scores), 1))


def lcs_length(a: List[str], b: List[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        lcs = lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append((2 * precision * recall) / (precision + recall))
    return float(sum(scores) / max(len(scores), 1))


def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()
    preds = preds.flatten(1)
    targets = targets.flatten(1)
    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)
    union = tp + fp + fn
    iou = (tp / union.clamp_min(1.0)).mean().item()
    pixp = (tp / (tp + fp).clamp_min(1.0)).mean().item()
    pixr = (tp / (tp + fn).clamp_min(1.0)).mean().item()
    pixf1 = ((2 * pixp * pixr) / max(pixp + pixr, 1e-8)) if (pixp + pixr) > 0 else 0.0
    return {"IoU": float(iou), "PixP": float(pixp), "PixR": float(pixr), "PixF1": float(pixf1)}


class MetricTracker:
    def __init__(self) -> None:
        self.cls_targets: List[torch.Tensor] = []
        self.cls_scores: List[torch.Tensor] = []
        self.cls_predictions: List[torch.Tensor] = []
        self.seg_logits: List[torch.Tensor] = []
        self.seg_targets: List[torch.Tensor] = []
        self.exp_predictions: List[str] = []
        self.exp_references: List[str] = []

    def update_classification(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
        preds = torch.argmax(logits, dim=1).detach().cpu()
        self.cls_scores.append(probs)
        self.cls_predictions.append(preds)
        self.cls_targets.append(targets.detach().cpu())

    def update_segmentation(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        self.seg_logits.append(logits.detach().cpu())
        self.seg_targets.append(targets.detach().cpu())

    def update_explanations(self, predictions: Iterable[str], references: Iterable[str]) -> None:
        self.exp_predictions.extend(list(predictions))
        self.exp_references.extend(list(references))

    def compute(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self.cls_targets:
            targets = torch.cat(self.cls_targets)
            scores = torch.cat(self.cls_scores)
            preds = torch.cat(self.cls_predictions)
            metrics["Acc"] = float((preds == targets).float().mean().item())
            metrics["AP"] = binary_average_precision(targets.float(), scores)
        if self.seg_logits:
            logits = torch.cat(self.seg_logits)
            targets = torch.cat(self.seg_targets)
            metrics.update(segmentation_metrics(logits, targets))
        if self.exp_predictions:
            metrics["BLEU"] = compute_bleu_1(self.exp_predictions, self.exp_references)
            metrics["ROUGE-L"] = compute_rouge_l(self.exp_predictions, self.exp_references)
        return metrics
