from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import random

from PIL import Image, ImageDraw, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm


def polygon_to_mask(polygons: List[List[float]], width: int, height: int) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(mask)
    for polygon in polygons:
        if len(polygon) < 6:
            continue
        points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
        drawer.polygon(points, outline=255, fill=255)
    return mask


def is_valid_image_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


class ImageNetClassificationDataset(Dataset):
    def __init__(
        self,
        roots: str | Path | Sequence[str | Path],
        split: str,
        image_size: int = 224,
        sample_limit_per_class: Optional[Dict[str, int]] = None,
        seed: int = 42,
    ) -> None:
        if isinstance(roots, (str, Path)):
            self.roots = [Path(roots)]
        else:
            self.roots = [Path(root) for root in roots]
        self.split = split
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.samples: List[Tuple[Path, int]] = []
        print(f"Preparing classification dataset split={split} from {len(self.roots)} root(s)...")
        class_map = {"nature": 0, "ai": 1}
        rng = random.Random(seed)
        for class_name, label in class_map.items():
            files: List[Path] = []
            for root in self.roots:
                class_dir = root / split / class_name
                if not class_dir.exists():
                    continue
                files.extend(p for p in class_dir.iterdir() if p.is_file())
            print(f"  Collected {len(files)} raw {class_name} files for split={split}.")
            validation_progress = tqdm(sorted(files), desc=f"validate {split} {class_name}", leave=False)
            valid_files = [p for p in validation_progress if is_valid_image_file(p)]
            print(f"  Kept {len(valid_files)} valid {class_name} files for split={split} before sampling.")
            if split == "train" and sample_limit_per_class is not None and class_name in sample_limit_per_class:
                rng.shuffle(valid_files)
                valid_files = valid_files[: sample_limit_per_class[class_name]]
                print(f"  Sampled {len(valid_files)} {class_name} files for split={split} using seed={seed}.")
            self.samples.extend((p, label) for p in valid_files)
        print(f"Finished classification dataset split={split}: total samples = {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | bool]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        return {
            "image": image_tensor,
            "cls_label": torch.tensor(label, dtype=torch.long),
            "has_cls": True,
            "has_loc": False,
            "has_exp": False,
            "image_path": str(image_path),
            "caption": "",
            "mask": torch.zeros((1, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32),
        }


class SynthScarsDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 256,
        indices: Optional[List[int]] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        annotation_path = self.root / split / "annotations" / f"{split}.json"
        with annotation_path.open("r", encoding="utf-8") as f:
            raw_entries = json.load(f)

        parsed_entries = []
        for item in raw_entries:
            key = next(iter(item.keys()))
            parsed_entries.append(item[key])

        if indices is not None:
            parsed_entries = [parsed_entries[i] for i in indices]

        self.entries = parsed_entries
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | bool]:
        entry = self.entries[index]
        image_path = self.root / self.split / "images" / entry["img_file_name"]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        polygons: List[List[float]] = []
        explanations: List[str] = []
        for ref in entry.get("refs", []):
            explanations.append(f"{ref.get('sentence', '')}: {ref.get('explanation', '')}".strip())
            polygons.extend(ref.get("segmentation", []))

        mask = polygon_to_mask(polygons, width, height)
        image_tensor = self.image_transform(image)
        mask_tensor = self.mask_transform(mask)
        caption = entry.get("caption", "")
        if explanations:
            caption = caption or " ".join(explanations)

        return {
            "image": image_tensor,
            "cls_label": torch.tensor(1, dtype=torch.long),
            "has_cls": True,
            "has_loc": True,
            "has_exp": True,
            "image_path": str(image_path),
            "caption": caption,
            "mask": (mask_tensor > 0.5).float(),
        }


class CourseClassificationDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 256) -> None:
        self.root = Path(root) / "任务一：检测"
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.samples: List[Tuple[Path, int]] = []
        class_map = {"0_real": 0, "1_fake": 1}
        for class_name, label in class_map.items():
            class_dir = self.root / class_name
            files = sorted([p for p in class_dir.iterdir() if p.is_file()])
            valid_files = [p for p in files if is_valid_image_file(p)]
            self.samples.extend((p, label) for p in valid_files)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | bool]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        return {
            "image": image_tensor,
            "cls_label": torch.tensor(label, dtype=torch.long),
            "has_cls": True,
            "has_loc": False,
            "has_exp": False,
            "image_path": str(image_path),
            "caption": "",
            "mask": torch.zeros((1, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32),
        }


class CourseExplanationDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 256) -> None:
        self.root = Path(root) / "任务二：解释"
        with (self.root / "explanation.json").open("r", encoding="utf-8") as f:
            raw_entries = json.load(f)

        self.entries = [entry for entry in raw_entries if is_valid_image_file(self.root / "images" / entry["img_file_name"])]
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | bool]:
        entry = self.entries[index]
        image_path = self.root / "images" / entry["img_file_name"]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        return {
            "image": image_tensor,
            "cls_label": torch.tensor(0, dtype=torch.long),
            "has_cls": False,
            "has_loc": False,
            "has_exp": True,
            "image_path": str(image_path),
            "caption": entry.get("caption", ""),
            "mask": torch.zeros((1, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32),
        }


class CourseLocalizationDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 256) -> None:
        self.root = Path(root) / "任务三：痕迹定位"
        with (self.root / "localization.json").open("r", encoding="utf-8") as f:
            raw_entries = json.load(f)

        self.entries = [entry for entry in raw_entries if is_valid_image_file(self.root / "images" / entry["img_file_name"])]
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | bool]:
        entry = self.entries[index]
        image_path = self.root / "images" / entry["img_file_name"]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        polygons: List[List[float]] = []
        for ref in entry.get("refs", []):
            polygons.extend(ref.get("segmentation", []))

        mask = polygon_to_mask(polygons, width, height)
        image_tensor = self.image_transform(image)
        mask_tensor = self.mask_transform(mask)
        return {
            "image": image_tensor,
            "cls_label": torch.tensor(0, dtype=torch.long),
            "has_cls": False,
            "has_loc": True,
            "has_exp": False,
            "image_path": str(image_path),
            "caption": "",
            "mask": (mask_tensor > 0.5).float(),
        }


def split_synthscars_indices(dataset_size: int, val_ratio: float, seed: int = 42) -> Tuple[List[int], List[int]]:
    indices = list(range(dataset_size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(dataset_size * val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices
