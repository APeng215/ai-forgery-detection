from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch


def load_metrics(path: str) -> Dict[str, float]:
    checkpoint = torch.load(path, map_location="cpu")
    metrics = checkpoint.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"Invalid checkpoint metrics format: {path}")
    return metrics


def format_value(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Path to baseline checkpoint")
    parser.add_argument("--candidate", required=True, help="Path to candidate/distilled checkpoint")
    parser.add_argument(
        "--keys",
        nargs="*",
        default=["Acc", "AP", "BLEU", "ROUGE-L", "IoU", "PixP", "PixR", "PixF1", "loss"],
        help="Metric keys to compare",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_path}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found: {candidate_path}")

    baseline = load_metrics(str(baseline_path))
    candidate = load_metrics(str(candidate_path))

    keys: List[str] = list(dict.fromkeys(args.keys))
    print(f"baseline: {baseline_path}")
    print(f"candidate: {candidate_path}")
    print("metric\tbaseline\tcandidate\tdelta(candidate-baseline)")
    for key in keys:
        b = baseline.get(key)
        c = candidate.get(key)
        delta = None
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            delta = float(c) - float(b)
        print(f"{key}\t{format_value(b) if isinstance(b, (int, float)) else '-'}\t{format_value(c) if isinstance(c, (int, float)) else '-'}\t{format_value(delta)}")


if __name__ == "__main__":
    main()
