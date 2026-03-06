#!/usr/bin/env python3
# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
LongBench 中文任务评测：Baseline vs TPTT

支持 multifieldqa_zh（单文档QA，F1 评分）。
数据从本地 JSONL 加载，适配 H800 无外网环境。

Usage:
    # Baseline only
    python scripts/longbench_tptt.py \
        --model /path/to/Qwen2.5-7B \
        --data /path/to/multifieldqa_zh.jsonl \
        --max-examples 200 --baseline-only

    # 对比 baseline vs TPTT checkpoint
    python scripts/longbench_tptt.py \
        --model /path/to/Qwen2.5-7B \
        --data /path/to/multifieldqa_zh.jsonl \
        --checkpoint /path/to/checkpoints/tptt_main \
        --max-examples 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from titans.eval_utils import generate_answer, load_baseline, load_tptt_model, score_example
from titans.tptt import reset_memory_states

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def load_longbench_data(jsonl_path: str, max_examples: int = 0) -> list[dict[str, Any]]:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            examples.append({
                "context": d.get("context", ""),
                "input": d.get("input", ""),
                "answers": d.get("answers", []),
                "length": d.get("length", 0),
            })
            if max_examples and len(examples) >= max_examples:
                break
    logger.info("加载 %d 条样本（来自 %s）", len(examples), path.name)
    return examples


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------


def build_prompt(context: str, question: str) -> str:
    return f"{context}\n\n{question}\n答："


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------


def evaluate(
    model: Any,
    tokenizer: Any,
    examples: list[dict],
    mode: str = "baseline",
    max_new_tokens: int = 64,
    max_ctx: int = 32768,
) -> dict[str, Any]:
    is_tptt = mode == "tptt"
    model_device = next(model.parameters()).device
    total_f1 = 0.0
    results = []

    for ex in tqdm(examples, desc=mode.upper()):
        if is_tptt:
            reset_memory_states(model)
        prompt = build_prompt(ex["context"], ex["input"])
        predicted = generate_answer(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens, max_ctx=max_ctx, device=model_device,
        )
        f1 = score_example(predicted, ex["answers"])
        total_f1 += f1
        results.append({
            "question": ex["input"],
            "predicted": predicted,
            "answers": ex["answers"],
            "f1": round(f1, 4),
        })

    avg_f1 = total_f1 / len(examples) if examples else 0.0
    return {"f1": avg_f1, "total": len(examples), "results": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LongBench 中文任务 TPTT 评测")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--data", required=True, help="本地 JSONL 文件路径")
    p.add_argument("--checkpoint", default=None,
                   help="TPTT checkpoint：.pt 文件路径 或 含 step_*.pt 的目录")
    p.add_argument("--num-memory-layers", type=int, default=1)
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-ctx", type=int, default=32768)
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--tptt-only", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    examples = load_longbench_data(args.data, args.max_examples)

    print()
    print("=== LongBench 中文 TPTT 评测 ===")
    print(f"模型:   {args.model}")
    print(f"数据:   {args.data}  ({len(examples)} 条)")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print()

    baseline_res = tptt_res = None

    if not args.tptt_only:
        model_bl, tokenizer = load_baseline(args.model, args.device, dtype)
        baseline_res = evaluate(
            model_bl, tokenizer, examples, mode="baseline",
            max_new_tokens=args.max_new_tokens, max_ctx=args.max_ctx,
        )
        print(f"Baseline F1: {baseline_res['f1']*100:.1f}%  ({baseline_res['total']} 条)")
        del model_bl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.baseline_only:
        model_tptt, tokenizer = load_tptt_model(
            args.model, args.device, dtype,
            checkpoint=args.checkpoint, num_memory_layers=args.num_memory_layers,
        )
        tptt_res = evaluate(
            model_tptt, tokenizer, examples, mode="tptt",
            max_new_tokens=args.max_new_tokens, max_ctx=args.max_ctx,
        )
        delta_str = ""
        if baseline_res is not None:
            delta = (tptt_res["f1"] - baseline_res["f1"]) * 100
            delta_str = f"  [{'+' if delta >= 0 else ''}{delta:.1f}pp]"
        ckpt_note = "(checkpoint)" if args.checkpoint else "(fresh memory)"
        print(f"TPTT {ckpt_note} F1: {tptt_res['f1']*100:.1f}%  ({tptt_res['total']} 条){delta_str}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "model": args.model,
            "checkpoint": args.checkpoint,
            "data": args.data,
            "num_examples": len(examples),
            "num_memory_layers": args.num_memory_layers,
        }
        if baseline_res:
            data["baseline"] = {
                "f1": baseline_res["f1"],
                "total": baseline_res["total"],
                "per_sample": baseline_res["results"],
            }
        if tptt_res:
            data["tptt"] = {
                "f1": tptt_res["f1"],
                "total": tptt_res["total"],
                "per_sample": tptt_res["results"],
            }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {out_path}")
    print()


if __name__ == "__main__":
    main()
