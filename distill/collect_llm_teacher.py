from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib import error, request

from torch.utils.data import ConcatDataset

from src.datasets.multitask_dataset import ImageNetClassificationDataset, SynthScarsDataset, split_synthscars_indices
from src.datasets.wrappers import DatasetWithSource
from src.training.utils import load_config, set_seed


def build_stage_b_train_dataset(cfg: Dict[str, Any]) -> ConcatDataset:
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

    train_limit = cfg["train"].get("synthscars_train_sample_limit")
    if train_limit is not None:
        train_indices = train_indices[: int(train_limit)]

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


def iter_image_paths(dataset: ConcatDataset) -> Iterable[str]:
    seen = set()
    for idx in range(len(dataset)):
        item = dataset[idx]
        image_path = str(item.get("image_path", ""))
        if not image_path or image_path in seen:
            continue
        seen.add(image_path)
        yield image_path


def image_to_data_url(image_path: str) -> str:
    data = Path(image_path).read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    suffix = Path(image_path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/png"
    return f"data:{mime};base64,{encoded}"


def build_messages(image_data_url: str) -> List[Dict[str, Any]]:
    system_prompt = (
        "You are an image forgery analysis assistant. "
        "Return ONLY a JSON object with keys: final_label, confidence, explanation_zh, evidence_points."
    )
    user_text = (
        "请判断图片是否为伪造图像，并给出中文解释。"
        "严格输出 JSON，字段如下："
        "{\"final_label\":\"real|fake\",\"confidence\":0~1,\"explanation_zh\":\"...\",\"evidence_points\":[\"...\",\"...\"]}."
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def call_openai_compatible(api_base: str, api_key: str, model: str, image_path: str, timeout: int) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": build_messages(image_to_data_url(image_path)),
        "temperature": 0.2,
    }
    body = json.dumps(payload).encode("utf-8")
    url = api_base.rstrip("/") + "/chat/completions"
    req = request.Request(url=url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")

    data = json.loads(raw)
    text = data["choices"][0]["message"]["content"].strip()

    # The model is instructed to return pure JSON. This fallback keeps automation robust.
    parsed = parse_json_content(text)
    return {"raw": text, "parsed": parsed}


def parse_json_content(text: str) -> Dict[str, Any]:
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        value = json.loads(candidate)
        if isinstance(value, dict):
            return value

    raise ValueError("Model content is not valid JSON object")


def call_openai_with_retry(
    api_base: str,
    api_key: str,
    model: str,
    image_path: str,
    timeout: int,
    max_retries: int,
    retry_initial_delay: float,
    retry_backoff: float,
    retry_max_delay: float,
) -> Dict[str, Any]:
    attempt = 0
    delay = max(0.0, retry_initial_delay)
    while True:
        try:
            return call_openai_compatible(
                api_base=api_base,
                api_key=api_key,
                model=model,
                image_path=image_path,
                timeout=timeout,
            )
        except Exception as exc:
            retryable = isinstance(exc, (TimeoutError, error.URLError, error.HTTPError, json.JSONDecodeError, ValueError))
            if not retryable or attempt >= max_retries:
                raise
            time.sleep(min(delay, retry_max_delay))
            delay = min(max(delay * retry_backoff, retry_initial_delay), retry_max_delay)
            attempt += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multitask_ema_distill.yaml")
    parser.add_argument("--output", default="distill/llm_teacher_outputs.jsonl")
    parser.add_argument("--model", required=True, help="OpenAI-compatible model name")
    parser.add_argument("--api-base", required=True, help="OpenAI-compatible API base URL, e.g. https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-initial-delay", type=float, default=1.0)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--retry-max-delay", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true", help="Skip image paths already present in output file")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env} is empty")

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])

    dataset = build_stage_b_train_dataset(cfg)
    print({"stage": "collect", "dataset_size": len(dataset), "limit": args.limit})
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    path = row.get("image_path", "")
                    if path:
                        done.add(path)
                except Exception:
                    continue

    processed = 0
    written = 0
    with output_path.open("a", encoding="utf-8") as f:
        for image_path in iter_image_paths(dataset):
            if args.limit > 0 and processed >= args.limit:
                break
            processed += 1
            if image_path in done:
                continue

            row: Dict[str, Any] = {"image_path": image_path}
            try:
                result = call_openai_with_retry(
                    api_base=args.api_base,
                    api_key=api_key,
                    model=args.model,
                    image_path=image_path,
                    timeout=args.timeout,
                    max_retries=max(0, args.max_retries),
                    retry_initial_delay=max(0.0, args.retry_initial_delay),
                    retry_backoff=max(1.0, args.retry_backoff),
                    retry_max_delay=max(0.1, args.retry_max_delay),
                )
                row["teacher"] = result["parsed"]
                row["raw"] = result["raw"]
                row["status"] = "ok"
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
                row["error_type"] = type(exc).__name__

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            written += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

            if args.log_every > 0 and written % args.log_every == 0:
                print({"processed": processed, "written": written, "output": str(output_path)})

    print({"processed": processed, "written": written, "output": str(output_path)})


if __name__ == "__main__":
    main()
