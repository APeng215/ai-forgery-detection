from __future__ import annotations

from pathlib import Path
import json

from PIL import Image

from src.datasets.multitask_dataset import polygon_to_mask


def export_synthscars_masks(root: str | Path, split: str) -> None:
    root = Path(root)
    annotation_path = root / split / "annotations" / f"{split}.json"
    output_dir = root / split / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    with annotation_path.open("r", encoding="utf-8") as f:
        raw_entries = json.load(f)

    for item in raw_entries:
        key = next(iter(item.keys()))
        entry = item[key]
        image_path = root / split / "images" / entry["img_file_name"]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        polygons = []
        for ref in entry.get("refs", []):
            polygons.extend(ref.get("segmentation", []))
        mask = polygon_to_mask(polygons, width, height)
        output_path = output_dir / f"{Path(entry['img_file_name']).stem}.png"
        mask.save(output_path)
