from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("[run]", " ".join(shlex.quote(x) for x in cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", required=True, help="OpenAI-compatible model name")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--config", default="configs/multitask_offline_llm_distill.yaml")
    parser.add_argument("--teacher-jsonl", default="distill/llm_teacher_outputs.jsonl")
    parser.add_argument("--teacher-cache", default="distill/teacher_cache_from_llm.pt")
    parser.add_argument("--output", default="outputs_llm_distill")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-initial-delay", type=float, default=1.0)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--retry-max-delay", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--baseline", default="", help="Optional baseline checkpoint for comparison")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    if not args.skip_collect and not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env} is empty")

    python = sys.executable
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"

    Path(args.teacher_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.teacher_cache).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if not args.skip_collect:
        collect_cmd = [
            python,
            "-m",
            "distill.collect_llm_teacher",
            "--config",
            args.config,
            "--api-base",
            args.api_base,
            "--model",
            args.model,
            "--api-key-env",
            args.api_key_env,
            "--output",
            args.teacher_jsonl,
            "--limit",
            str(args.limit),
            "--sleep",
            str(args.sleep),
            "--max-retries",
            str(args.max_retries),
            "--retry-initial-delay",
            str(args.retry_initial_delay),
            "--retry-backoff",
            str(args.retry_backoff),
            "--retry-max-delay",
            str(args.retry_max_delay),
        ]
        if args.resume:
            collect_cmd.append("--resume")
        run_cmd(collect_cmd, env=env)

    build_cache_cmd = [
        python,
        "-m",
        "distill.build_cache_from_llm",
        "--input",
        args.teacher_jsonl,
        "--output",
        args.teacher_cache,
        "--config",
        args.config,
    ]
    run_cmd(build_cache_cmd, env=env)

    if not args.skip_train:
        train_cmd = [
            python,
            "train_multitask.py",
            "--config",
            args.config,
            "--output",
            args.output,
        ]
        if args.allow_cpu:
            train_cmd.append("--allow-cpu")
        run_cmd(train_cmd, env=env)

    if args.baseline:
        compare_cmd = [
            python,
            "-m",
            "distill.compare_checkpoints",
            "--baseline",
            args.baseline,
            "--candidate",
            str(Path(args.output) / "best.pt"),
        ]
        run_cmd(compare_cmd, env=env)

    print(
        {
            "teacher_jsonl": args.teacher_jsonl,
            "teacher_cache": args.teacher_cache,
            "train_output": args.output,
            "baseline": args.baseline or None,
        }
    )


if __name__ == "__main__":
    main()
