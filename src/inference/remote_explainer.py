from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Mapping
from urllib import error, request

from src.training.utils import get_env


def _local_response(local_explanation: str, source: str, fallback_reason: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "explanation": local_explanation,
        "source": source,
        "evidence_points": [],
        "confidence": None,
        "need_human_review": False,
    }
    if fallback_reason:
        payload["fallback_reason"] = fallback_reason
    return payload


def _encode_image_as_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _build_prompt(local_result: Mapping[str, Any]) -> str:
    return (
        "You are the explanation enhancement module for an image forgery detection system. "
        "The inputs include the original image, a suspicious-region mask, and the local model's prediction. "
        "Do not overturn the local classification result. Generate an English explanation that closely follows this course annotation style. "
        "Use this exact two-part template: "
        "'Upon examining the image. I have found: <one concise overall sentence>. "
        "To elaborate, I have found the following artifacts. <artifact name>:<artifact detail>. <artifact name>:<artifact detail>.' "
        "Keep the wording concrete and image-grounded, and prefer visible object-level artifacts over abstract forensic language. "
        "Mention specific body parts, objects, regions, or boundaries when possible. "
        "Do not write bullet lists inside the explanation. Do not mention the mask explicitly. "
        "Return a JSON object only, with no markdown fences and no extra text. "
        "The JSON must contain: explanation (using the exact template above), evidence_points (2-4 short plain English strings without leading dashes or bullets), confidence (a number from 0 to 1), and need_human_review (a boolean).\n\n"
        f"Local label: {local_result['label']}\n"
        f"Local fake_score: {float(local_result['fake_score']):.4f}\n"
        f"Local explanation: {local_result['explanation']}\n"
        "The second image is the local suspicious-region mask, where brighter areas indicate more suspicious regions."
    )


def _extract_text_content(response_body: Mapping[str, Any]) -> str:
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Remote response does not contain choices")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise ValueError("Remote response does not contain message")
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
        if text_parts:
            return "".join(text_parts)
    raise ValueError("Remote response content is not supported")


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _normalize_payload(payload: Mapping[str, Any], local_explanation: str, source: str) -> dict[str, Any]:
    explanation = payload.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("Remote response missing explanation")

    evidence_points_raw = payload.get("evidence_points", [])
    evidence_points: list[str] = []
    if isinstance(evidence_points_raw, list):
        evidence_points = [str(item).strip() for item in evidence_points_raw if str(item).strip()]

    confidence = payload.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("Remote response confidence is invalid") from exc

    need_human_review = payload.get("need_human_review", False)
    if not isinstance(need_human_review, bool):
        need_human_review = bool(need_human_review)

    return {
        "explanation": explanation.strip(),
        "source": source,
        "evidence_points": evidence_points,
        "confidence": confidence,
        "need_human_review": need_human_review,
        "local_explanation": local_explanation,
    }


def _get_api_key(remote_cfg: Mapping[str, Any]) -> str | None:
    raw_api_key = remote_cfg.get("api_key")
    if raw_api_key is not None:
        api_key = str(raw_api_key).strip()
        if api_key:
            return api_key

    api_key_env = str(remote_cfg.get("api_key_env", "DASHSCOPE_API_KEY"))
    return get_env(api_key_env)


def enhance_explanation(
    image_path: str | Path,
    mask_path: str | Path,
    local_result: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    inference_cfg = cfg.get("inference") or {}
    remote_cfg = inference_cfg.get("remote_explanation") or {}
    local_explanation = str(local_result["explanation"])
    if not remote_cfg.get("enabled", False):
        return _local_response(local_explanation, source="local")

    fallback_to_local = bool(remote_cfg.get("fallback_to_local", True))
    api_key_env = str(remote_cfg.get("api_key_env", "DASHSCOPE_API_KEY"))
    api_key = _get_api_key(remote_cfg)
    if not api_key:
        fallback = _local_response(
            local_explanation,
            source="local_fallback",
            fallback_reason=f"Missing remote_explanation.api_key and environment variable: {api_key_env}",
        )
        if fallback_to_local:
            return fallback
        raise RuntimeError(fallback["fallback_reason"])

    api_base = str(remote_cfg.get("api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1")).rstrip("/")
    model = str(remote_cfg.get("model", "qwen3-vl-plus"))
    timeout_sec = float(remote_cfg.get("timeout_sec", 20))

    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You improve English explanations for an image forgery detection pipeline. Return JSON only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_prompt(local_result)},
                    {"type": "image_url", "image_url": {"url": _encode_image_as_data_url(image_path)}},
                    {"type": "image_url", "image_url": {"url": _encode_image_as_data_url(mask_path)}},
                ],
            },
        ],
        "temperature": 0.2,
    }

    req = request.Request(
        url=f"{api_base}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_sec) as response:
            response_body = json.loads(response.read().decode("utf-8"))
        raw_content = _extract_text_content(response_body)
        parsed = json.loads(_strip_code_fences(raw_content))
        if not isinstance(parsed, Mapping):
            raise ValueError("Remote response JSON is not an object")
        return _normalize_payload(parsed, local_explanation=local_explanation, source="remote")
    except (error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        fallback = _local_response(local_explanation, source="local_fallback", fallback_reason=str(exc))
        if fallback_to_local:
            return fallback
        raise RuntimeError(str(exc)) from exc
