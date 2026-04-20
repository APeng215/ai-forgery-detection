from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from train_multitask import build_candidate_bank, collate_fn, evaluate, get_training_device
from src.datasets.multitask_dataset import (
    CourseClassificationDataset,
    CourseExplanationDataset,
    CourseLocalizationDataset,
    SynthScarsDataset,
)
from src.datasets.wrappers import DatasetWithSource
from src.inference import evaluate_with_inference_policy, get_inference_mode
from src.models.multitask_model import MultiTaskForgeryModel
from src.training.losses import TextFeatureEncoder, build_multitask_losses
from src.training.utils import load_config


def evaluate_synthscars_test(cfg, model, losses, text_encoder, device):
    dataset = DatasetWithSource(
        SynthScarsDataset(cfg["data"]["synthscars_root"], split="test", image_size=cfg["model"]["image_size_seg"]),
        "synthscars",
    )
    dataloader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], collate_fn=collate_fn)
    candidate_texts, candidate_features = build_candidate_bank(dataset.dataset, text_encoder, device)
    inference_mode = get_inference_mode(cfg)
    if inference_mode == "plan_b":
        return evaluate_with_inference_policy(model, dataloader, losses, text_encoder, candidate_texts, candidate_features, device, cfg, stage_label="eval", epoch_label="test")
    return evaluate(model, dataloader, losses, text_encoder, candidate_texts, candidate_features, device, cfg, stage_label="eval", epoch_label="test")


def evaluate_course(cfg, model, losses, text_encoder, device):
    course_root = cfg["data"]["course_root"]
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"]["num_workers"]
    image_size = cfg["model"]["image_size_seg"]

    candidate_dataset = SynthScarsDataset(cfg["data"]["synthscars_root"], split="train", image_size=image_size)
    candidate_texts, candidate_features = build_candidate_bank(candidate_dataset, text_encoder, device)

    cls_dataset = DatasetWithSource(CourseClassificationDataset(course_root, image_size=image_size), "course_cls")
    exp_dataset = DatasetWithSource(CourseExplanationDataset(course_root, image_size=image_size), "course_exp")
    loc_dataset = DatasetWithSource(CourseLocalizationDataset(course_root, image_size=image_size), "course_loc")

    cls_loader = DataLoader(cls_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    exp_loader = DataLoader(exp_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    loc_loader = DataLoader(loc_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    inference_mode = get_inference_mode(cfg)
    if inference_mode == "plan_b":
        cls_metrics = evaluate_with_inference_policy(model, cls_loader, losses, text_encoder, candidate_texts, candidate_features, device, cfg, stage_label="course-cls", epoch_label="test")
        exp_metrics = evaluate_with_inference_policy(model, exp_loader, losses, text_encoder, candidate_texts, candidate_features, device, cfg, stage_label="course-exp", epoch_label="test")
    else:
        cls_metrics = evaluate(model, cls_loader, losses, text_encoder, device=device, cfg=cfg, candidate_texts=candidate_texts, candidate_features=candidate_features, stage_label="course-cls", epoch_label="test")
        exp_metrics = evaluate(model, exp_loader, losses, text_encoder, device=device, cfg=cfg, candidate_texts=candidate_texts, candidate_features=candidate_features, stage_label="course-exp", epoch_label="test")
    loc_metrics = evaluate(model, loc_loader, losses, text_encoder, device=device, cfg=cfg, candidate_texts=candidate_texts, candidate_features=candidate_features, stage_label="course-loc", epoch_label="test")

    result = {
        "classification": cls_metrics,
        "explanation": exp_metrics,
        "localization": loc_metrics,
        "Acc": cls_metrics.get("Acc", 0.0),
        "AP": cls_metrics.get("AP", 0.0),
        "BLEU": exp_metrics.get("BLEU", 0.0),
        "ROUGE-L": exp_metrics.get("ROUGE-L", 0.0),
        "IoU": loc_metrics.get("IoU", 0.0),
        "PixP": loc_metrics.get("PixP", 0.0),
        "PixR": loc_metrics.get("PixR", 0.0),
        "PixF1": loc_metrics.get("PixF1", 0.0),
    }
    if inference_mode == "plan_b":
        result["plan_b_policy"] = {
            "classification": cls_metrics.get("policy", {}),
            "explanation": exp_metrics.get("policy", {}),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multitask.yaml")
    parser.add_argument("--checkpoint", default="outputs/best.pt")
    parser.add_argument("--target", choices=["synthscars_test", "course"], default="synthscars_test")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_training_device(allow_cpu=args.allow_cpu)

    model = MultiTaskForgeryModel(pretrained_backbone=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    losses = build_multitask_losses()
    text_encoder = TextFeatureEncoder(feature_dim=cfg["model"]["explanation_feature_dim"]).to(device)

    if args.target == "course":
        metrics = evaluate_course(cfg, model, losses, text_encoder, device)
    else:
        metrics = evaluate_synthscars_test(cfg, model, losses, text_encoder, device)
    print(metrics)


if __name__ == "__main__":
    main()
