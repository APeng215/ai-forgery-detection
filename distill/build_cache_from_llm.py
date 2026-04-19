from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from src.training.losses import TextFeatureEncoder
from src.training.utils import load_config


def label_to_fake_prob(teacher: Dict[str, Any]) -> float:
    if "fake_score" in teacher and isinstance(teacher["fake_score"], (int, float)):
        p = float(teacher["fake_score"])
        return max(1e-4, min(1.0 - 1e-4, p))

    label = str(teacher.get("final_label", "")).strip().lower()
    confidence = teacher.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    confidence = float(confidence)
    confidence = max(0.0, min(1.0, confidence))

    if label == "fake":
        p_fake = confidence
    elif label == "real":
        p_fake = 1.0 - confidence
    else:
        p_fake = 0.5
    return max(1e-4, min(1.0 - 1e-4, p_fake))


def probs_to_logits(p_fake: float) -> torch.Tensor:
    probs = torch.tensor([1.0 - p_fake, p_fake], dtype=torch.float32)
    return torch.log(probs)


def extract_explanation(teacher: Dict[str, Any]) -> str:
    text = teacher.get("explanation_zh", "")
    if isinstance(text, str) and text.strip():
        return text.strip()
    points = teacher.get("evidence_points", [])
    if isinstance(points, list):
        merged = "；".join([str(x).strip() for x in points if str(x).strip()])
        if merged:
            return merged
    return ""


def parse_line(line: str) -> Tuple[str, Dict[str, Any] | None]:
    row = json.loads(line)
    image_path = str(row.get("image_path", "")).strip()
    if not image_path:
        return "", None
    if row.get("status") != "ok":
        return image_path, None
    teacher = row.get("teacher")
    if not isinstance(teacher, dict):
        return image_path, None
    return image_path, teacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="distill/llm_teacher_outputs.jsonl")
    parser.add_argument("--output", default="distill/teacher_cache_from_llm.pt")
    parser.add_argument("--config", default="configs/multitask_ema_distill.yaml")
    parser.add_argument("--feature-dim", type=int, default=0, help="0 means reading from config")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    cfg = load_config(args.config)
    feature_dim = args.feature_dim if args.feature_dim > 0 else int(cfg["model"]["explanation_feature_dim"])
    encoder = TextFeatureEncoder(feature_dim=feature_dim)

    samples: Dict[str, Dict[str, torch.Tensor]] = {}
    total = 0
    valid = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                image_path, teacher = parse_line(line)
            except Exception:
                continue
            if not image_path or teacher is None:
                continue

            p_fake = label_to_fake_prob(teacher)
            logits = probs_to_logits(p_fake)

            explanation = extract_explanation(teacher)
            exp_feature = encoder.encode([explanation], device=torch.device("cpu"))[0].cpu()

            samples[image_path] = {
                "logits": logits,
                "explanation_features": exp_feature,
            }
            valid += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "samples": samples,
            "meta": {
                "source": str(input_path),
                "total_rows": total,
                "valid_rows": valid,
                "feature_dim": feature_dim,
            },
        },
        output_path,
    )

    print({"input": str(input_path), "output": str(output_path), "total": total, "valid": valid})


if __name__ == "__main__":
    main()
