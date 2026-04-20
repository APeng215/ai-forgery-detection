from __future__ import annotations

import io
from typing import Any, Iterable, Mapping

from PIL import Image
import torch
from tqdm.auto import tqdm

from train_multitask import compute_batch_loss
from src.training.metrics import binary_average_precision, compute_bleu_1, compute_rouge_l, segmentation_metrics

from .remote_explainer import enhance_explanation, rejudge_hard_case


VALID_INFERENCE_MODES = {"local", "enhance", "plan_b"}


def get_inference_mode(cfg: Mapping[str, Any]) -> str:
    inference_cfg = cfg.get("inference") or {}
    mode = str(inference_cfg.get("mode", "")).strip().lower()
    if not mode:
        remote_cfg = inference_cfg.get("remote_explanation") or {}
        return "enhance" if remote_cfg.get("enabled", False) else "local"
    if mode not in VALID_INFERENCE_MODES:
        raise ValueError(f"Unsupported inference.mode: {mode}")
    return mode


def get_classification_threshold(cfg: Mapping[str, Any]) -> float:
    inference_cfg = cfg.get("inference") or {}
    value = float(inference_cfg.get("classification_threshold", 0.5))
    return min(max(value, 0.0), 1.0)


def get_plan_b_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    inference_cfg = cfg.get("inference") or {}
    return inference_cfg.get("plan_b") or {}


def mask_tensor_to_png_bytes(mask: torch.Tensor) -> bytes:
    image = Image.fromarray((mask.detach().cpu().clamp(0.0, 1.0) * 255).to(torch.uint8).numpy(), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_local_results(
    logits: torch.Tensor,
    mask_logits: torch.Tensor,
    explanations: Iterable[str],
    cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    cls_probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu()
    mask_probs = torch.sigmoid(mask_logits.detach())[:, 0].cpu()
    threshold = get_classification_threshold(cfg)

    local_results: list[dict[str, Any]] = []
    for cls_prob, explanation, mask_prob in zip(cls_probs.tolist(), explanations, mask_probs):
        label = "fake" if cls_prob >= threshold else "real"
        mask_area_ratio = float((mask_prob >= 0.5).float().mean().item())
        local_results.append(
            {
                "label": label,
                "fake_score": float(cls_prob),
                "explanation": str(explanation),
                "mask_area_ratio": mask_area_ratio,
            }
        )
    return local_results, mask_probs


def should_route_plan_b(local_result: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[bool, list[str]]:
    plan_b_cfg = get_plan_b_cfg(cfg)
    threshold = get_classification_threshold(cfg)
    fake_score = float(local_result["fake_score"])
    label = str(local_result["label"])
    mask_area_ratio = float(local_result.get("mask_area_ratio", 0.0))

    reasons: list[str] = []
    score_margin = max(0.0, float(plan_b_cfg.get("score_margin", 0.0)))
    if abs(fake_score - threshold) <= score_margin:
        reasons.append("near_threshold")

    if bool(plan_b_cfg.get("route_on_mask_disagreement", True)):
        fake_mask_area_min = max(0.0, float(plan_b_cfg.get("fake_mask_area_min", 0.01)))
        real_mask_area_max = max(0.0, float(plan_b_cfg.get("real_mask_area_max", 0.10)))
        if label == "fake" and mask_area_ratio < fake_mask_area_min:
            reasons.append("fake_low_mask_support")
        if label == "real" and mask_area_ratio > real_mask_area_max:
            reasons.append("real_high_mask_support")

    return bool(reasons), reasons


def _base_final_result(local_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": str(local_result["label"]),
        "fake_score": float(local_result["fake_score"]),
        "explanation": str(local_result["explanation"]),
        "decision_source": "local",
        "explanation_source": "local",
        "routed_to_remote": False,
        "routing_reasons": [],
        "local_label": str(local_result["label"]),
        "local_fake_score": float(local_result["fake_score"]),
        "local_explanation": str(local_result["explanation"]),
        "mask_area_ratio": float(local_result.get("mask_area_ratio", 0.0)),
        "need_human_review": False,
    }


def resolve_inference_result(
    image_path: str,
    mask_prob: torch.Tensor,
    local_result: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    mode = get_inference_mode(cfg)
    final_result = _base_final_result(local_result)

    if mode == "local":
        return final_result

    mask_bytes = mask_tensor_to_png_bytes(mask_prob)

    if mode == "enhance":
        remote_result = enhance_explanation(image_path, mask_bytes, local_result, cfg)
        final_result["explanation"] = remote_result["explanation"]
        final_result["explanation_source"] = remote_result["source"]
        if remote_result.get("evidence_points"):
            final_result["evidence_points"] = remote_result["evidence_points"]
        if remote_result.get("confidence") is not None:
            final_result["explanation_confidence"] = remote_result["confidence"]
        if "need_human_review" in remote_result:
            final_result["need_human_review"] = remote_result["need_human_review"]
        if remote_result.get("fallback_reason"):
            final_result["fallback_reason"] = remote_result["fallback_reason"]
        if remote_result.get("source") == "remote":
            final_result["decision_source"] = "local_enhanced"
        return final_result

    routed, routing_reasons = should_route_plan_b(local_result, cfg)
    final_result["routed_to_remote"] = routed
    final_result["routing_reasons"] = routing_reasons
    if not routed:
        final_result["decision_source"] = "plan_b_local"
        return final_result

    remote_result = rejudge_hard_case(image_path, mask_bytes, local_result, routing_reasons, cfg)
    if remote_result.get("fallback_reason"):
        final_result["fallback_reason"] = remote_result["fallback_reason"]

    if remote_result.get("source") != "remote_rejudge":
        final_result["decision_source"] = "plan_b_fallback_local"
        final_result["need_human_review"] = bool(remote_result.get("need_human_review", False))
        return final_result

    remote_label = str(remote_result["remote_label"])
    remote_fake_score = float(remote_result["remote_fake_score"])
    remote_confidence = float(remote_result.get("confidence") or 0.0)
    remote_agrees = remote_label == final_result["local_label"]
    override_threshold = max(0.0, float(get_plan_b_cfg(cfg).get("remote_override_confidence_min", 0.65)))

    final_result["remote_label"] = remote_label
    final_result["remote_fake_score"] = remote_fake_score
    final_result["remote_explanation"] = remote_result["explanation"]
    final_result["remote_confidence"] = remote_confidence
    final_result["remote_agrees"] = remote_agrees
    if remote_result.get("evidence_points"):
        final_result["evidence_points"] = remote_result["evidence_points"]
    if remote_result.get("decision_rationale"):
        final_result["decision_rationale"] = remote_result["decision_rationale"]

    if remote_agrees:
        final_result["label"] = remote_label
        final_result["fake_score"] = remote_fake_score
        final_result["explanation"] = remote_result["explanation"]
        final_result["decision_source"] = "plan_b_remote_agree"
        final_result["explanation_source"] = "remote_rejudge"
        final_result["need_human_review"] = bool(remote_result.get("need_human_review", False))
        return final_result

    if remote_confidence >= override_threshold:
        final_result["label"] = remote_label
        final_result["fake_score"] = remote_fake_score
        final_result["explanation"] = remote_result["explanation"]
        final_result["decision_source"] = "plan_b_remote_override"
        final_result["explanation_source"] = "remote_rejudge"
        final_result["need_human_review"] = bool(remote_result.get("need_human_review", False))
        return final_result

    final_result["decision_source"] = "plan_b_local_after_disagreement"
    final_result["need_human_review"] = True
    return final_result


def _bump_counter(bucket: dict[str, int], key: str) -> None:
    bucket[key] = bucket.get(key, 0) + 1


def make_policy_stats() -> dict[str, Any]:
    return {
        "total_cases": 0,
        "routed_cases": 0,
        "remote_response_cases": 0,
        "override_cases": 0,
        "fallback_cases": 0,
        "local_after_disagreement_cases": 0,
        "remote_agreement_cases": 0,
        "remote_disagreement_cases": 0,
        "human_review_cases": 0,
        "routing_reason_counts": {},
        "decision_source_counts": {},
        "fallback_reason_counts": {},
    }


def update_policy_stats(stats: dict[str, Any], final_result: Mapping[str, Any]) -> None:
    stats["total_cases"] += 1

    decision_source = str(final_result.get("decision_source") or "unknown")
    _bump_counter(stats["decision_source_counts"], decision_source)

    if final_result.get("routed_to_remote", False):
        stats["routed_cases"] += 1
    for reason in final_result.get("routing_reasons") or []:
        _bump_counter(stats["routing_reason_counts"], str(reason))

    remote_agrees = final_result.get("remote_agrees")
    if remote_agrees is True:
        stats["remote_response_cases"] += 1
        stats["remote_agreement_cases"] += 1
    elif remote_agrees is False:
        stats["remote_response_cases"] += 1
        stats["remote_disagreement_cases"] += 1

    if decision_source == "plan_b_remote_override":
        stats["override_cases"] += 1
    if decision_source == "plan_b_fallback_local":
        stats["fallback_cases"] += 1
    if decision_source == "plan_b_local_after_disagreement":
        stats["local_after_disagreement_cases"] += 1

    if final_result.get("need_human_review"):
        stats["human_review_cases"] += 1

    fallback_reason = final_result.get("fallback_reason")
    if fallback_reason:
        _bump_counter(stats["fallback_reason_counts"], str(fallback_reason))


def finalize_policy_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    total_cases = int(stats["total_cases"])
    routed_cases = int(stats["routed_cases"])
    remote_response_cases = int(stats["remote_response_cases"])
    remote_agreement_cases = int(stats["remote_agreement_cases"])

    remote_agreement_rate = remote_agreement_cases / remote_response_cases if remote_response_cases > 0 else 0.0
    routing_rate = routed_cases / total_cases if total_cases > 0 else 0.0
    remote_response_rate_over_routed = remote_response_cases / routed_cases if routed_cases > 0 else 0.0
    override_rate_over_routed = int(stats["override_cases"]) / routed_cases if routed_cases > 0 else 0.0
    fallback_rate_over_routed = int(stats["fallback_cases"]) / routed_cases if routed_cases > 0 else 0.0
    human_review_rate = int(stats["human_review_cases"]) / total_cases if total_cases > 0 else 0.0

    return {
        **stats,
        "routing_rate": routing_rate,
        "remote_response_rate_over_routed": remote_response_rate_over_routed,
        "remote_agreement_rate": remote_agreement_rate,
        "override_rate_over_routed": override_rate_over_routed,
        "fallback_rate_over_routed": fallback_rate_over_routed,
        "human_review_rate": human_review_rate,
    }


def evaluate_with_inference_policy(model, dataloader, losses, text_encoder, candidate_texts, candidate_features, device, cfg, stage_label: str, epoch_label: str):
    model.eval()
    total_loss = 0.0
    steps = 0
    cls_targets: list[torch.Tensor] = []
    cls_scores: list[torch.Tensor] = []
    cls_predictions: list[torch.Tensor] = []
    seg_logits: list[torch.Tensor] = []
    seg_targets: list[torch.Tensor] = []
    exp_predictions: list[str] = []
    exp_references: list[str] = []
    policy_stats = make_policy_stats()

    progress = tqdm(dataloader, desc=f"{stage_label} val {epoch_label}", leave=False)
    with torch.no_grad():
        for batch in progress:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            outputs = model(batch["image"])
            loss, scalars = compute_batch_loss(batch, outputs, losses, text_encoder, device, cfg)
            total_loss += float(loss.item())
            steps += 1
            progress.set_postfix(loss=f"{loss.item():.4f}", **{k: f"{v:.4f}" for k, v in scalars.items()})

            explanations = model.explanation_head.predict(outputs["explanation_features"], candidate_texts, candidate_features)
            local_results, mask_probs = build_local_results(outputs["logits"], outputs["mask_logits"], explanations, cfg)
            final_results: list[dict[str, Any]] = []
            for image_path, mask_prob, local_result in zip(batch["image_path"], mask_probs, local_results):
                final_result = resolve_inference_result(str(image_path), mask_prob, local_result, cfg)
                final_results.append(final_result)
                update_policy_stats(policy_stats, final_result)

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                routed=policy_stats["routed_cases"],
                override=policy_stats["override_cases"],
                fallback=policy_stats["fallback_cases"],
                review=policy_stats["human_review_cases"],
            )

            if batch["has_cls"].any():
                cls_mask = batch["has_cls"].detach().cpu().tolist()
                targets: list[int] = []
                scores: list[float] = []
                preds: list[int] = []
                for idx, has_cls in enumerate(cls_mask):
                    if not has_cls:
                        continue
                    targets.append(int(batch["cls_label"][idx].detach().cpu().item()))
                    scores.append(float(final_results[idx]["fake_score"]))
                    preds.append(1 if final_results[idx]["label"] == "fake" else 0)
                if targets:
                    cls_targets.append(torch.tensor(targets, dtype=torch.long))
                    cls_scores.append(torch.tensor(scores, dtype=torch.float32))
                    cls_predictions.append(torch.tensor(preds, dtype=torch.long))

            if batch["has_loc"].any():
                loc_mask = batch["has_loc"]
                seg_logits.append(outputs["mask_logits"][loc_mask].detach().cpu())
                seg_targets.append(batch["mask"][loc_mask].detach().cpu())

            if batch["has_exp"].any():
                exp_mask = batch["has_exp"].detach().cpu().tolist()
                for idx, has_exp in enumerate(exp_mask):
                    if not has_exp:
                        continue
                    exp_predictions.append(str(final_results[idx]["explanation"]))
                    exp_references.append(str(batch["caption"][idx]))

    metrics: dict[str, Any] = {}
    if cls_targets:
        targets_tensor = torch.cat(cls_targets)
        scores_tensor = torch.cat(cls_scores)
        preds_tensor = torch.cat(cls_predictions)
        metrics["Acc"] = float((preds_tensor == targets_tensor).float().mean().item())
        metrics["AP"] = binary_average_precision(targets_tensor.float(), scores_tensor)
    if seg_logits:
        logits_tensor = torch.cat(seg_logits)
        targets_tensor = torch.cat(seg_targets)
        metrics.update(segmentation_metrics(logits_tensor, targets_tensor))
    if exp_predictions:
        metrics["BLEU"] = compute_bleu_1(exp_predictions, exp_references)
        metrics["ROUGE-L"] = compute_rouge_l(exp_predictions, exp_references)
    metrics["loss"] = total_loss / max(steps, 1)
    metrics["policy"] = finalize_policy_stats(policy_stats)
    return metrics
