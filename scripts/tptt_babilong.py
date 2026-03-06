#!/usr/bin/env python3
# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
TPTT × Qwen2.5-7B × BABILong 评测脚本

baseline:  原始 Qwen2.5-7B，无 memory 注入
tptt:      注入 NeuralLongTermMemory，可加载 checkpoint

Usage:
    # 快速验证（无 checkpoint）
    uv run python scripts/tptt_babilong.py \\
        --model Qwen/Qwen2.5-7B \\
        --task qa1 --context-length 16k \\
        --max-examples 20 --baseline-only

    # 完整对比
    uv run python scripts/tptt_babilong.py \\
        --task qa1 --context-length 16k \\
        --max-examples 100 --output results/tptt_qa1_16k.json

    # 加载 checkpoint（.pt 文件或含 step_*.pt 的目录均可）
    uv run python scripts/tptt_babilong.py \\
        --task qa1 --context-length 16k \\
        --checkpoint checkpoints/tptt_main/step_0005000.pt \\
        --max-examples 100 --output results/tptt_trained_qa1_16k.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from titans.eval_utils import generate_answer, load_baseline, load_tptt_model
from titans.tptt import reset_memory_states

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BABILong 常量
# ---------------------------------------------------------------------------

BABILONG_TASKS = {
    "qa1": "single-supporting-fact",
    "qa2": "two-supporting-facts",
    "qa3": "three-supporting-facts",
    "qa4": "two-arg-relations",
    "qa5": "three-arg-relations",
    "qa6": "yes-no-questions",
    "qa7": "counting",
    "qa8": "lists-sets",
    "qa9": "simple-negation",
    "qa10": "indefinite-knowledge",
}

CONTEXT_LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def load_babilong_task(
    task: str,
    context_length: str = "0k",
    split: str = "test",
    local_data_dir: str | None = None,
) -> list[dict[str, Any]]:
    task_name = BABILONG_TASKS.get(task)
    if not task_name:
        raise ValueError(f"未知任务: {task}，可选: {list(BABILONG_TASKS.keys())}")

    # 优先读本地 JSON（格式：<local_data_dir>/<task>/<context_length>.json）
    if local_data_dir is not None:
        import json as _json
        local_path = Path(local_data_dir) / task / f"{context_length}.json"
        if not local_path.exists():
            raise FileNotFoundError(f"本地数据文件不存在: {local_path}")
        logger.info("从本地加载 BABILong %s @ %s: %s", task, context_length, local_path)
        with open(local_path, encoding="utf-8") as f:
            data = _json.load(f)
        return [
            {
                "input": item.get("input", item.get("context", "")),
                "question": item.get("question", ""),
                "answer": item.get("answer", item.get("target", "")),
            }
            for item in data
        ]

    if not HAS_DATASETS:
        raise ImportError("需要 datasets 库：pip install datasets")

    logger.info("加载 BABILong %s (%s) @ %s", task, task_name, context_length)
    dataset = load_dataset("RMT-team/babilong", f"{task_name}_{context_length}", split=split)
    return [
        {
            "input": item.get("input", item.get("context", "")),
            "question": item.get("question", ""),
            "answer": item.get("answer", item.get("target", "")),
        }
        for item in dataset
    ]


# ---------------------------------------------------------------------------
# 推理工具
# ---------------------------------------------------------------------------


def build_prompt(context: str, question: str) -> str:
    return f"{context}\n\nQuestion: {question}\nAnswer:"


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------


def evaluate(
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    mode: str = "baseline",
    max_new_tokens: int = 32,
    max_ctx: int = 32768,
) -> dict[str, Any]:
    is_tptt = mode == "tptt"
    model_device = next(model.parameters()).device
    correct = 0
    results = []

    for example in tqdm(examples, desc=mode.upper()):
        if is_tptt:
            reset_memory_states(model)

        prompt = build_prompt(example["input"], example["question"])
        predicted = generate_answer(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens, max_ctx=max_ctx, device=model_device,
        )

        pred_norm = normalize_answer(predicted)
        gold_norm = normalize_answer(example["answer"])
        is_correct = gold_norm in pred_norm or pred_norm in gold_norm
        if is_correct:
            correct += 1

        results.append({
            "question": example["question"],
            "predicted": predicted,
            "gold": example["answer"],
            "correct": is_correct,
        })

    accuracy = correct / len(examples) if examples else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": len(examples), "results": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TPTT × Qwen2.5 × BABILong 评测")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--checkpoint", default=None,
                   help="TPTT checkpoint：.pt 文件路径 或 含 step_*.pt 的目录")
    p.add_argument("--num-memory-layers", type=int, default=1)
    p.add_argument("--task", default="qa1", choices=list(BABILONG_TASKS.keys()))
    p.add_argument("--context-length", default="16k", choices=CONTEXT_LENGTHS)
    p.add_argument("--max-examples", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-ctx", type=int, default=32768)
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--tptt-only", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--output", default=None)
    p.add_argument("--local-data-dir", default=None, metavar="DIR",
                   help="本地 BABILong 数据目录（<dir>/<task>/<ctx>.json）")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，退回 CPU")
        args.device = "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    examples = load_babilong_task(
        args.task, args.context_length, local_data_dir=args.local_data_dir,
    )
    if args.max_examples:
        examples = examples[: args.max_examples]
    logger.info("共 %d 个样本", len(examples))

    print()
    print("=== TPTT BABILong 评测 ===")
    print(f"模型:   {args.model}")
    print(f"任务:   {args.task} @ {args.context_length}")
    print(f"样本数: {len(examples)}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print()

    baseline_results = tptt_results = None

    if not args.tptt_only:
        model_bl, tokenizer = load_baseline(args.model, args.device, dtype)
        baseline_results = evaluate(
            model_bl, tokenizer, examples, mode="baseline",
            max_new_tokens=args.max_new_tokens, max_ctx=args.max_ctx,
        )
        bl_acc = baseline_results["accuracy"] * 100
        print(f"Baseline: {bl_acc:.1f}%  ({baseline_results['correct']}/{baseline_results['total']})")
        del model_bl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.baseline_only:
        model_tptt, tokenizer = load_tptt_model(
            args.model, args.device, dtype,
            checkpoint=args.checkpoint, num_memory_layers=args.num_memory_layers,
        )
        tptt_results = evaluate(
            model_tptt, tokenizer, examples, mode="tptt",
            max_new_tokens=args.max_new_tokens, max_ctx=args.max_ctx,
        )
        tp_acc = tptt_results["accuracy"] * 100
        delta_str = ""
        if baseline_results is not None:
            delta = tp_acc - baseline_results["accuracy"] * 100
            delta_str = f"  [{'+' if delta >= 0 else ''}{delta:.1f}pp]"
        ckpt_note = "(checkpoint)" if args.checkpoint else "(fresh memory)"
        print(f"TPTT {ckpt_note}: {tp_acc:.1f}%  ({tptt_results['correct']}/{tptt_results['total']}){delta_str}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "model": args.model,
            "checkpoint": args.checkpoint,
            "task": args.task,
            "context_length": args.context_length,
            "num_examples": len(examples),
            "num_memory_layers": args.num_memory_layers,
        }
        if baseline_results:
            data["baseline"] = {k: v for k, v in baseline_results.items() if k != "results"}
            data["baseline"]["per_sample"] = baseline_results["results"]
        if tptt_results:
            data["tptt"] = {k: v for k, v in tptt_results.items() if k != "results"}
            data["tptt"]["per_sample"] = tptt_results["results"]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {out_path}")
    print()


if __name__ == "__main__":
    main()
