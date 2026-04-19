from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any, Dict

import yaml


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(current, value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    local_path = config_path.with_name(f"{config_path.stem}.local{config_path.suffix}")
    if local_path != config_path and local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            local_config = yaml.safe_load(f) or {}
        if isinstance(local_config, dict):
            config = _merge_dicts(config, local_config)

    return config


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
