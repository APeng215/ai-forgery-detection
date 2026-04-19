from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from src.datasets.multitask_dataset import ImageNetClassificationDataset, SynthScarsDataset, split_synthscars_indices
from src.datasets.wrappers import DatasetWithSource
from src.models.multitask_model import MultiTaskForgeryModel
from src.training.utils import load_config, set_seed


def collate_for_cache(batch):
    images = torch.stack([item["image"] for item in batch])
    image_paths = [item.get("image_path", "") for item in batch]
    return {"image": images, "image_path": image_paths}


def build_stage_b_train_dataset(cfg):
    image_size = cfg["model"]["image_size_seg"]
    seed = cfg["train"]["seed"]
    sample_limits = {
        "ai": cfg["train"]["imagenet_fake_sample_limit"],
        "nature": cfg["train"]["imagenet_real_sample_limit"],
    }

    imagenet = DatasetWithSource(
        ImageNetClassificationDataset(
            cfg["data"]["imagenet_roots"],
            split="train",
            image_size=image_size,
            sample_limit_per_class=sample_limits,
            seed=seed,
        ),
        "imagenet",
    )

    synth_full = SynthScarsDataset(cfg["data"]["synthscars_root"], split="train", image_size=image_size)
    train_indices, _ = split_synthscars_indices(len(synth_full), cfg["train"]["synthscars_val_ratio"], seed)
    synth = DatasetWithSource(
        SynthScarsDataset(
            cfg["data"]["synthscars_root"],
            split="train",
            image_size=image_size,
            indices=train_indices,
        ),
        "synthscars",
    )

    return ConcatDataset([imagenet, synth])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multitask.yaml")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output", default="distill/teacher_cache.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Use --allow-cpu to run cache generation on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskForgeryModel(pretrained_backbone=False).to(device)
    checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = build_stage_b_train_dataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_for_cache,
    )

    samples = {}
    progress = tqdm(dataloader, desc="build teacher cache", leave=True)
    with torch.no_grad():
        for batch in progress:
            images = batch["image"].to(device)
            outputs = model(images)
            logits = outputs["logits"].detach().cpu()
            exp_features = outputs["explanation_features"].detach().cpu()
            for idx, path in enumerate(batch["image_path"]):
                if not path:
                    continue
                samples[path] = {
                    "logits": logits[idx],
                    "explanation_features": exp_features[idx],
                }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": samples}, output_path)
    print({"cache_path": str(output_path), "num_samples": len(samples)})


if __name__ == "__main__":
    main()
