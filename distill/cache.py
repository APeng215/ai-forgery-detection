from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch


class OfflineTeacherCache:
    """Read-only cache of teacher outputs keyed by sample identifier."""

    def __init__(self, cache_path: str | Path, strict: bool = False) -> None:
        self.cache_path = Path(cache_path)
        self.strict = strict
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Teacher cache file not found: {self.cache_path}")
        raw = torch.load(self.cache_path, map_location="cpu")
        if isinstance(raw, dict) and "samples" in raw and isinstance(raw["samples"], dict):
            self.samples = raw["samples"]
        elif isinstance(raw, dict):
            self.samples = raw
        else:
            raise ValueError("Teacher cache format is invalid. Expected a dict or {'samples': dict}.")

    def __len__(self) -> int:
        return len(self.samples)

    def get(self, key: str) -> Optional[Dict[str, torch.Tensor]]:
        sample = self.samples.get(key)
        if sample is None:
            if self.strict:
                raise KeyError(f"Missing teacher outputs for key: {key}")
            return None
        if not isinstance(sample, dict):
            raise ValueError(f"Teacher cache entry must be dict, got: {type(sample)} for key: {key}")

        result: Dict[str, torch.Tensor] = {}
        if "logits" in sample and sample["logits"] is not None:
            result["logits"] = torch.as_tensor(sample["logits"], dtype=torch.float32)
        if "explanation_features" in sample and sample["explanation_features"] is not None:
            result["explanation_features"] = torch.as_tensor(sample["explanation_features"], dtype=torch.float32)
        return result

    def get_many(self, keys: List[str]) -> List[Optional[Dict[str, torch.Tensor]]]:
        return [self.get(key) for key in keys]
