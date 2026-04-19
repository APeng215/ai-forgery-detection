from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

from src.datasets.multitask_dataset import SynthScarsDataset
from src.inference import enhance_explanation
from src.models.multitask_model import MultiTaskForgeryModel
from src.training.losses import TextFeatureEncoder
from src.training.utils import load_config


def build_candidate_bank(dataset: SynthScarsDataset, text_encoder: TextFeatureEncoder, device: torch.device):
    captions = []
    for entry in dataset.entries:
        caption = entry.get("caption", "")
        if caption:
            captions.append(caption)
    captions = list(dict.fromkeys(captions))
    features = text_encoder.encode(captions, device)
    return captions, features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--config", default="configs/multitask.yaml")
    parser.add_argument("--checkpoint", default="outputs/best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(args.image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((cfg["model"]["image_size_seg"], cfg["model"]["image_size_seg"])),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    model = MultiTaskForgeryModel(pretrained_backbone=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    text_encoder = TextFeatureEncoder(feature_dim=cfg["model"]["explanation_feature_dim"]).to(device)
    synth_dataset = SynthScarsDataset(cfg["data"]["synthscars_root"], split="train", image_size=cfg["model"]["image_size_seg"])
    candidate_texts, candidate_features = build_candidate_bank(synth_dataset, text_encoder, device)

    with torch.no_grad():
        outputs = model(image_tensor)
        cls_prob = torch.softmax(outputs["logits"], dim=1)[0, 1].item()
        label = "fake" if cls_prob >= 0.5 else "real"
        explanation = model.explanation_head.predict(outputs["explanation_features"], candidate_texts, candidate_features)[0]
        mask = torch.sigmoid(outputs["mask_logits"])[0, 0].cpu()

    output_mask_path = Path(args.image_path).with_name(f"{Path(args.image_path).stem}_pred_mask.png")
    mask_image = transforms.ToPILImage()(mask)
    mask_image.save(output_mask_path)

    local_result = {
        "label": label,
        "fake_score": cls_prob,
        "explanation": explanation,
        "mask_path": str(output_mask_path),
    }
    remote_result = enhance_explanation(args.image_path, output_mask_path, local_result, cfg)

    final_result = {
        "label": label,
        "fake_score": cls_prob,
        "explanation": remote_result["explanation"],
        "mask_path": str(output_mask_path),
        "explanation_source": remote_result["source"],
    }
    if remote_result.get("evidence_points"):
        final_result["evidence_points"] = remote_result["evidence_points"]
    if remote_result.get("confidence") is not None:
        final_result["explanation_confidence"] = remote_result["confidence"]
    if "need_human_review" in remote_result:
        final_result["need_human_review"] = remote_result["need_human_review"]
    if remote_result.get("source") == "remote":
        final_result["local_explanation"] = explanation
    if remote_result.get("fallback_reason"):
        final_result["fallback_reason"] = remote_result["fallback_reason"]

    print(final_result)


if __name__ == "__main__":
    main()
