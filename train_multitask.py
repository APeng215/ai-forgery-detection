from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from src.datasets.multitask_dataset import ImageNetClassificationDataset, SynthScarsDataset, split_synthscars_indices
from src.datasets.wrappers import DatasetWithSource
from src.models.multitask_model import MultiTaskForgeryModel
from src.training.losses import TextFeatureEncoder, build_multitask_losses
from src.training.metrics import MetricTracker
from src.training.utils import load_config, set_seed


def collate_fn(batch: List[Dict]) -> Dict:
    images = torch.stack([item["image"] for item in batch])
    cls_labels = torch.stack([item["cls_label"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    captions = [item["caption"] for item in batch]
    has_cls = torch.tensor([item["has_cls"] for item in batch], dtype=torch.bool)
    has_loc = torch.tensor([item["has_loc"] for item in batch], dtype=torch.bool)
    has_exp = torch.tensor([item["has_exp"] for item in batch], dtype=torch.bool)
    sources = [item.get("source", "unknown") for item in batch]
    return {
        "image": images,
        "cls_label": cls_labels,
        "mask": masks,
        "caption": captions,
        "has_cls": has_cls,
        "has_loc": has_loc,
        "has_exp": has_exp,
        "source": sources,
    }


def get_training_device(allow_cpu: bool = False) -> torch.device:
    if not torch.cuda.is_available():
        if allow_cpu:
            print("Using CPU because --allow-cpu was provided.")
            return torch.device("cpu")
        raise RuntimeError(
            "CUDA is not available in the current Python environment. "
            "Install a CUDA-enabled PyTorch build before training."
        )
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    props = torch.cuda.get_device_properties(device)
    total_gb = props.total_memory / (1024 ** 3)
    print(f"Using GPU: {props.name} ({total_gb:.1f} GB)")
    return device


def compute_batch_loss(batch: Dict, outputs: Dict, losses, text_encoder, device: torch.device, cfg: Dict) -> tuple[torch.Tensor, Dict[str, float]]:
    total_loss = torch.tensor(0.0, device=device)
    scalars: Dict[str, float] = {}

    if batch["has_cls"].any():
        cls_mask = batch["has_cls"]
        cls_loss = losses.cls_loss(outputs["logits"][cls_mask], batch["cls_label"][cls_mask])
        total_loss = total_loss + cfg["train"]["cls_weight"] * cls_loss
        scalars["cls_loss"] = float(cls_loss.item())

    if batch["has_loc"].any():
        loc_mask = batch["has_loc"]
        bce_loss = losses.loc_bce_loss(outputs["mask_logits"][loc_mask], batch["mask"][loc_mask])
        dice_loss = losses.loc_dice_loss(outputs["mask_logits"][loc_mask], batch["mask"][loc_mask])
        loc_loss = bce_loss + dice_loss
        total_loss = total_loss + cfg["train"]["loc_weight"] * loc_loss
        scalars["loc_loss"] = float(loc_loss.item())

    if batch["has_exp"].any():
        exp_mask = batch["has_exp"]
        references = [batch["caption"][i] for i, flag in enumerate(exp_mask.tolist()) if flag]
        target_features = text_encoder.encode(references, device)
        exp_loss = losses.exp_loss(outputs["explanation_features"][exp_mask], target_features)
        total_loss = total_loss + cfg["train"]["exp_weight"] * exp_loss
        scalars["exp_loss"] = float(exp_loss.item())

    return total_loss, scalars


def build_candidate_bank(dataset: SynthScarsDataset, text_encoder: TextFeatureEncoder, device: torch.device) -> tuple[list[str], torch.Tensor]:
    captions = []
    for entry in dataset.entries:
        caption = entry.get("caption", "")
        if caption:
            captions.append(caption)
    unique_captions = list(dict.fromkeys(captions))
    features = text_encoder.encode(unique_captions, device)
    return unique_captions, features


def evaluate(model, dataloader, losses, text_encoder, candidate_texts, candidate_features, device, cfg, stage_label: str, epoch_label: str):
    model.eval()
    tracker = MetricTracker()
    total_loss = 0.0
    steps = 0
    progress = tqdm(dataloader, desc=f"{stage_label} val {epoch_label}", leave=False)
    with torch.no_grad():
        for batch in progress:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            outputs = model(batch["image"])
            loss, scalars = compute_batch_loss(batch, outputs, losses, text_encoder, device, cfg)
            total_loss += float(loss.item())
            steps += 1
            progress.set_postfix(loss=f"{loss.item():.4f}", **{k: f"{v:.4f}" for k, v in scalars.items()})
            if batch["has_cls"].any():
                cls_mask = batch["has_cls"]
                tracker.update_classification(outputs["logits"][cls_mask], batch["cls_label"][cls_mask])
            if batch["has_loc"].any():
                loc_mask = batch["has_loc"]
                tracker.update_segmentation(outputs["mask_logits"][loc_mask], batch["mask"][loc_mask])
            if batch["has_exp"].any():
                exp_mask = batch["has_exp"]
                predicted = model.explanation_head.predict(outputs["explanation_features"][exp_mask], candidate_texts, candidate_features)
                references = [batch["caption"][i] for i, flag in enumerate(exp_mask.tolist()) if flag]
                tracker.update_explanations(predicted, references)
    metrics = tracker.compute()
    metrics["loss"] = total_loss / max(steps, 1)
    return metrics


def run_epoch(model, dataloader, optimizer, scaler, losses, text_encoder, device, cfg, stage_label: str, epoch_index: int, total_epochs: int):
    model.train()
    loss_history = []
    source_counter = Counter()
    progress = tqdm(dataloader, desc=f"{stage_label} train {epoch_index}/{total_epochs}", leave=True)
    for batch in progress:
        source_counter.update(batch["source"])
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=cfg["train"]["mixed_precision"] and device.type == "cuda"):
            outputs = model(batch["image"])
            loss, scalars = compute_batch_loss(batch, outputs, losses, text_encoder, device, cfg)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_history.append(float(loss.item()))
        progress.set_postfix(loss=f"{loss.item():.4f}", **{k: f"{v:.4f}" for k, v in scalars.items()})
    return sum(loss_history) / max(len(loss_history), 1), dict(source_counter)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multitask.yaml")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])

    device = get_training_device(allow_cpu=args.allow_cpu)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = cfg["model"]["image_size_seg"]
    imagenet_roots = cfg["data"]["imagenet_roots"]
    synth_root = cfg["data"]["synthscars_root"]
    print(f"Loading config from {args.config}")
    print(f"Classification roots: {imagenet_roots}")
    print(f"SynthScars root: {synth_root}")
    print(f"Output directory: {output_dir}")
    sample_limits = {
        "ai": cfg["train"]["imagenet_fake_sample_limit"],
        "nature": cfg["train"]["imagenet_real_sample_limit"],
    }

    print("Building stage A classification datasets...")
    stage_a_train = DatasetWithSource(
        ImageNetClassificationDataset(
            imagenet_roots,
            split="train",
            image_size=image_size,
            sample_limit_per_class=sample_limits,
            seed=cfg["train"]["seed"],
        ),
        "imagenet",
    )
    stage_a_val = DatasetWithSource(
        ImageNetClassificationDataset(
            imagenet_roots,
            split="val",
            image_size=image_size,
            sample_limit_per_class=None,
            seed=cfg["train"]["seed"],
        ),
        "imagenet",
    )
    print(f"Loaded stage A datasets: train={len(stage_a_train)} val={len(stage_a_val)}")

    print("Loading SynthScars datasets...")
    synth_full = SynthScarsDataset(synth_root, split="train", image_size=image_size)
    train_indices, val_indices = split_synthscars_indices(len(synth_full), cfg["train"]["synthscars_val_ratio"], cfg["train"]["seed"])
    synth_train = DatasetWithSource(SynthScarsDataset(synth_root, split="train", image_size=image_size, indices=train_indices), "synthscars")
    synth_val = DatasetWithSource(SynthScarsDataset(synth_root, split="train", image_size=image_size, indices=val_indices), "synthscars")
    print(f"Loaded SynthScars datasets: full={len(synth_full)} train={len(synth_train)} val={len(synth_val)}")

    print("Creating dataloaders...")
    stage_a_train_loader = DataLoader(stage_a_train, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], collate_fn=collate_fn)
    stage_a_val_loader = DataLoader(stage_a_val, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], collate_fn=collate_fn)

    joint_train = ConcatDataset([stage_a_train, synth_train])
    joint_val = ConcatDataset([stage_a_val, synth_val])
    joint_train_loader = DataLoader(joint_train, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], collate_fn=collate_fn)
    joint_val_loader = DataLoader(joint_val, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], collate_fn=collate_fn)
    print(f"Created dataloaders: stage_a_train_batches={len(stage_a_train_loader)} stage_a_val_batches={len(stage_a_val_loader)} joint_train_batches={len(joint_train_loader)} joint_val_batches={len(joint_val_loader)}")

    print("Building model and optimizer...")
    model = MultiTaskForgeryModel(pretrained_backbone=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.amp.GradScaler(device="cuda", enabled=cfg["train"]["mixed_precision"] and device.type == "cuda")
    losses = build_multitask_losses()
    text_encoder = TextFeatureEncoder(feature_dim=cfg["model"]["explanation_feature_dim"]).to(device)
    print("Building explanation candidate bank...")
    candidate_texts, candidate_features = build_candidate_bank(synth_train.dataset, text_encoder, device)
    print(f"Built explanation candidate bank with {len(candidate_texts)} texts.")

    best_score = float("-inf")

    print("Starting stage A training...")
    for epoch in range(cfg["train"]["epochs_stage_a"]):
        train_loss, sources = run_epoch(model, stage_a_train_loader, optimizer, scaler, losses, text_encoder, device, cfg, stage_label="A", epoch_index=epoch + 1, total_epochs=cfg["train"]["epochs_stage_a"])
        metrics = evaluate(model, stage_a_val_loader, losses, text_encoder, candidate_texts, candidate_features, device, cfg, stage_label="A", epoch_label=f"{epoch + 1}/{cfg['train']['epochs_stage_a']}")
        checkpoint = {"epoch": epoch + 1, "stage": "A", "model_state": model.state_dict(), "config": cfg, "metrics": metrics}
        torch.save(checkpoint, output_dir / "latest.pt")
        score = metrics.get("Acc", 0.0) + metrics.get("AP", 0.0)
        if score > best_score:
            best_score = score
            torch.save(checkpoint, output_dir / "best.pt")
        print({"stage": "A", "epoch": epoch + 1, "train_loss": train_loss, "metrics": metrics, "sources": sources})

    print("Starting stage B joint training...")
    for epoch in range(cfg["train"]["epochs_stage_b"]):
        train_loss, sources = run_epoch(model, joint_train_loader, optimizer, scaler, losses, text_encoder, device, cfg, stage_label="B", epoch_index=epoch + 1, total_epochs=cfg["train"]["epochs_stage_b"])
        metrics = evaluate(model, joint_val_loader, losses, text_encoder, candidate_texts, candidate_features, device, cfg, stage_label="B", epoch_label=f"{epoch + 1}/{cfg['train']['epochs_stage_b']}")
        checkpoint = {"epoch": epoch + 1, "stage": "B", "model_state": model.state_dict(), "config": cfg, "metrics": metrics}
        torch.save(checkpoint, output_dir / "latest.pt")
        score = metrics.get("Acc", 0.0) + metrics.get("AP", 0.0) + metrics.get("BLEU", 0.0) + metrics.get("ROUGE-L", 0.0) + metrics.get("IoU", 0.0) + metrics.get("PixF1", 0.0)
        if score > best_score:
            best_score = score
            torch.save(checkpoint, output_dir / "best.pt")
        print({"stage": "B", "epoch": epoch + 1, "train_loss": train_loss, "metrics": metrics, "sources": sources})


if __name__ == "__main__":
    main()
