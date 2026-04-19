from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from src.datasets.multitask_dataset import ImageNetClassificationDataset, SynthScarsDataset, split_synthscars_indices
from src.datasets.wrappers import DatasetWithSource
from distill.interface import DistillProvider, compute_online_distill_loss, init_ema_teacher, update_ema_teacher
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
    image_paths = [item.get("image_path", "") for item in batch]
    return {
        "image": images,
        "cls_label": cls_labels,
        "mask": masks,
        "caption": captions,
        "has_cls": has_cls,
        "has_loc": has_loc,
        "has_exp": has_exp,
        "source": sources,
        "image_path": image_paths,
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


def compute_batch_loss(batch: Dict, outputs: Dict, losses, text_encoder, device: torch.device, cfg: Dict, distill_provider: DistillProvider | None = None, stage_label: str = "", distill_cfg: Dict | None = None, teacher_outputs: Dict[str, torch.Tensor] | None = None) -> tuple[torch.Tensor, Dict[str, float]]:
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

    if distill_provider is not None:
        distill_loss, distill_scalars = distill_provider.compute(outputs=outputs, batch=batch, stage_label=stage_label)
        total_loss = total_loss + distill_loss
        scalars.update(distill_scalars)

    if distill_cfg is not None and bool(distill_cfg.get("enabled", False)):
        mode = distill_cfg.get("mode", "offline_cache")
        stage_b_only = bool(distill_cfg.get("stage_b_only", True))
        if mode == "ema_online" and (not stage_b_only or stage_label == "B"):
            online_distill_loss, online_scalars = compute_online_distill_loss(
                outputs=outputs,
                teacher_outputs=teacher_outputs,
                batch=batch,
                device=device,
                config=distill_cfg,
            )
            total_loss = total_loss + online_distill_loss
            scalars.update(online_scalars)

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


def run_epoch(model, dataloader, optimizer, scaler, losses, text_encoder, device, cfg, stage_label: str, epoch_index: int, total_epochs: int, distill_provider: DistillProvider | None = None, teacher_model=None, distill_cfg: Dict | None = None):
    model.train()
    loss_history = []
    source_counter = Counter()
    distill_scalar_sum: Dict[str, float] = defaultdict(float)
    distill_scalar_count: Dict[str, int] = defaultdict(int)
    progress = tqdm(dataloader, desc=f"{stage_label} train {epoch_index}/{total_epochs}", leave=True)
    for batch in progress:
        source_counter.update(batch["source"])
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        teacher_outputs = None
        if teacher_model is not None and distill_cfg is not None and bool(distill_cfg.get("enabled", False)):
            mode = distill_cfg.get("mode", "offline_cache")
            stage_b_only = bool(distill_cfg.get("stage_b_only", True))
            if mode == "ema_online" and (not stage_b_only or stage_label == "B"):
                with torch.no_grad():
                    teacher_outputs = teacher_model(batch["image"])
        with torch.autocast(device_type=device.type, enabled=cfg["train"]["mixed_precision"] and device.type == "cuda"):
            outputs = model(batch["image"])
            loss, scalars = compute_batch_loss(
                batch,
                outputs,
                losses,
                text_encoder,
                device,
                cfg,
                distill_provider=distill_provider,
                stage_label=stage_label,
                distill_cfg=distill_cfg,
                teacher_outputs=teacher_outputs,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if teacher_model is not None and distill_cfg is not None and bool(distill_cfg.get("enabled", False)):
            mode = distill_cfg.get("mode", "offline_cache")
            stage_b_only = bool(distill_cfg.get("stage_b_only", True))
            if mode == "ema_online" and (not stage_b_only or stage_label == "B"):
                update_ema_teacher(teacher_model, model, decay=float(distill_cfg.get("ema_decay", 0.999)))
        for key, value in scalars.items():
            if key.startswith("distill"):
                distill_scalar_sum[key] += float(value)
                distill_scalar_count[key] += 1
        loss_history.append(float(loss.item()))
        progress.set_postfix(loss=f"{loss.item():.4f}", **{k: f"{v:.4f}" for k, v in scalars.items()})
    distill_epoch_stats = {
        key: distill_scalar_sum[key] / max(distill_scalar_count[key], 1)
        for key in distill_scalar_sum
    }
    return sum(loss_history) / max(len(loss_history), 1), dict(source_counter), distill_epoch_stats


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
    val_sample_limits = None
    val_fake_limit = cfg["train"].get("imagenet_val_fake_sample_limit")
    val_real_limit = cfg["train"].get("imagenet_val_real_sample_limit")
    if val_fake_limit is not None or val_real_limit is not None:
        val_sample_limits = {}
        if val_fake_limit is not None:
            val_sample_limits["ai"] = int(val_fake_limit)
        if val_real_limit is not None:
            val_sample_limits["nature"] = int(val_real_limit)

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
            sample_limit_per_class=val_sample_limits,
            seed=cfg["train"]["seed"],
        ),
        "imagenet",
    )
    print(f"Loaded stage A datasets: train={len(stage_a_train)} val={len(stage_a_val)}")

    print("Loading SynthScars datasets...")
    synth_full = SynthScarsDataset(synth_root, split="train", image_size=image_size)
    train_indices, val_indices = split_synthscars_indices(len(synth_full), cfg["train"]["synthscars_val_ratio"], cfg["train"]["seed"])
    synth_train_sample_limit = cfg["train"].get("synthscars_train_sample_limit")
    if synth_train_sample_limit is not None:
        train_indices = train_indices[: int(synth_train_sample_limit)]
    synth_val_sample_limit = cfg["train"].get("synthscars_val_sample_limit")
    if synth_val_sample_limit is not None:
        val_indices = val_indices[: int(synth_val_sample_limit)]
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
    distill_cfg = cfg.get("distill", {})
    distill_mode = distill_cfg.get("mode", "offline_cache")
    distill_provider = None
    teacher_model = None
    if bool(distill_cfg.get("enabled", False)):
        if distill_mode == "offline_cache":
            distill_provider = DistillProvider(distill_cfg, device)
            print("Distillation enabled: true (offline_cache)")
            if distill_provider.cache is not None:
                print(f"Teacher cache loaded: {len(distill_provider.cache)} entries")
        elif distill_mode == "ema_online":
            teacher_model = init_ema_teacher(model, device)
            print("Distillation enabled: true (ema_online)")
        else:
            raise ValueError(f"Unsupported distill mode: {distill_mode}")
    else:
        print("Distillation enabled: false")
    print("Building explanation candidate bank...")
    candidate_texts, candidate_features = build_candidate_bank(synth_train.dataset, text_encoder, device)
    print(f"Built explanation candidate bank with {len(candidate_texts)} texts.")

    best_score = float("-inf")

    print("Starting stage A training...")
    for epoch in range(cfg["train"]["epochs_stage_a"]):
        train_loss, sources, distill_epoch_stats = run_epoch(
            model,
            stage_a_train_loader,
            optimizer,
            scaler,
            losses,
            text_encoder,
            device,
            cfg,
            stage_label="A",
            epoch_index=epoch + 1,
            total_epochs=cfg["train"]["epochs_stage_a"],
            distill_provider=distill_provider,
            teacher_model=teacher_model,
            distill_cfg=distill_cfg,
        )
        metrics = evaluate(model, stage_a_val_loader, losses, text_encoder, candidate_texts, candidate_features, device, cfg, stage_label="A", epoch_label=f"{epoch + 1}/{cfg['train']['epochs_stage_a']}")
        checkpoint = {"epoch": epoch + 1, "stage": "A", "model_state": model.state_dict(), "config": cfg, "metrics": metrics}
        torch.save(checkpoint, output_dir / "latest.pt")
        score = metrics.get("Acc", 0.0) + metrics.get("AP", 0.0)
        if score > best_score:
            best_score = score
            torch.save(checkpoint, output_dir / "best.pt")
        summary = {"stage": "A", "epoch": epoch + 1, "train_loss": train_loss, "metrics": metrics, "sources": sources}
        if distill_epoch_stats:
            summary["distill"] = distill_epoch_stats
        print(summary)

    print("Starting stage B joint training...")
    for epoch in range(cfg["train"]["epochs_stage_b"]):
        train_loss, sources, distill_epoch_stats = run_epoch(
            model,
            joint_train_loader,
            optimizer,
            scaler,
            losses,
            text_encoder,
            device,
            cfg,
            stage_label="B",
            epoch_index=epoch + 1,
            total_epochs=cfg["train"]["epochs_stage_b"],
            distill_provider=distill_provider,
            teacher_model=teacher_model,
            distill_cfg=distill_cfg,
        )
        metrics = evaluate(model, joint_val_loader, losses, text_encoder, candidate_texts, candidate_features, device, cfg, stage_label="B", epoch_label=f"{epoch + 1}/{cfg['train']['epochs_stage_b']}")
        checkpoint = {"epoch": epoch + 1, "stage": "B", "model_state": model.state_dict(), "config": cfg, "metrics": metrics}
        torch.save(checkpoint, output_dir / "latest.pt")
        score = metrics.get("Acc", 0.0) + metrics.get("AP", 0.0) + metrics.get("BLEU", 0.0) + metrics.get("ROUGE-L", 0.0) + metrics.get("IoU", 0.0) + metrics.get("PixF1", 0.0)
        if score > best_score:
            best_score = score
            torch.save(checkpoint, output_dir / "best.pt")
        summary = {"stage": "B", "epoch": epoch + 1, "train_loss": train_loss, "metrics": metrics, "sources": sources}
        if distill_epoch_stats:
            summary["distill"] = distill_epoch_stats
        print(summary)


if __name__ == "__main__":
    main()
