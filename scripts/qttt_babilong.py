#!/usr/bin/env python3
# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
qTTT × Qwen3-7B × BABILong 评测脚本

实现路径 A（qTTT）的快速验证实验。核心思路来自 arXiv:2512.13898：
推理时只更新 W_Q（query projection），用文档内随机 span 做 next-token prediction
作为训练信号，无需架构改动，验证"推理时学习"对长文档问答的收益。

目标：在 BABILong 上对比 Qwen3-7B baseline vs. qTTT，测试 16k/32k 两个长度。

Usage:
    # 只跑 baseline，确认环境
    uv run python scripts/qttt_babilong.py \\
        --model Qwen/Qwen3-7B \\
        --task qa1 --context-length 16k \\
        --max-examples 20 --baseline-only

    # 完整对比（baseline + qTTT）
    uv run python scripts/qttt_babilong.py \\
        --task qa1 --context-length 16k \\
        --max-examples 100 --num-steps 20 \\
        --output results/qttt_qa1_16k.json

    # 跑 32k 看长度影响
    uv run python scripts/qttt_babilong.py \\
        --task qa1 --context-length 32k \\
        --max-examples 50 --num-steps 20 \\
        --output results/qttt_qa1_32k.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Optional imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# BABILong 任务列表
# =============================================================================

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


# =============================================================================
# qTTT 配置
# =============================================================================


@dataclass
class QTTTConfig:
    num_steps: int = 20
    lr: float = 1e-4
    span_len: int = 512
    batch_size: int = 4
    max_ctx: int = 32768


# =============================================================================
# 模型加载
# =============================================================================


def load_model(
    model_name: str,
    device: torch.device,
) -> tuple[Any, Any]:
    """加载 HuggingFace 因果语言模型和分词器。

    Args:
        model_name: HuggingFace 模型名称（如 Qwen/Qwen3-7B）
        device: 目标设备

    Returns:
        (model, tokenizer) 元组
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("需要 transformers 库：pip install transformers")

    logger.info(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    logger.info("模型加载完成")
    return model, tokenizer


# =============================================================================
# W_Q 权重保存 / 恢复
# =============================================================================


def save_q_weights(model: Any) -> dict[str, torch.Tensor]:
    """保存所有 q_proj 层的权重快照。

    Args:
        model: HuggingFace 模型

    Returns:
        {参数名: 权重克隆} 字典
    """
    saved = {}
    for name, param in model.named_parameters():
        if "q_proj" in name:
            saved[name] = param.data.clone()
    return saved


def restore_q_weights(model: Any, saved: dict[str, torch.Tensor]) -> None:
    """将 q_proj 权重恢复到保存时的值。

    Args:
        model: HuggingFace 模型
        saved: save_q_weights 返回的字典
    """
    for name, param in model.named_parameters():
        if name in saved:
            param.data.copy_(saved[name])


# =============================================================================
# qTTT 更新
# =============================================================================


def qttt_update(
    model: Any,
    doc_ids: torch.Tensor,
    config: QTTTConfig,
) -> list[float]:
    """在文档上执行 qTTT 更新（只训练 W_Q）。

    随机采样文档内的 span，用 next-token prediction 作为训练信号，
    只对 q_proj 参数执行梯度更新。

    Args:
        model: HuggingFace 模型（已 eval()）
        doc_ids: 文档 token IDs，shape [1, L]，在正确 device 上
        config: qTTT 配置

    Returns:
        每步的 loss 值列表
    """
    doc_len = doc_ids.shape[1]

    # 只对 q_proj 开启梯度
    for name, param in model.named_parameters():
        param.requires_grad = "q_proj" in name

    q_params = [p for n, p in model.named_parameters() if "q_proj" in n]
    if not q_params:
        logger.warning("未找到 q_proj 参数，跳过 qTTT 更新")
        return []

    optimizer = torch.optim.AdamW(q_params, lr=config.lr)
    model.train()

    losses = []
    span_len = config.span_len

    # 文档太短时退化为单 span
    max_start = max(1, doc_len - span_len)

    for step in range(config.num_steps):
        # 随机采样 batch_size 个 span
        starts = [random.randint(0, max_start - 1) for _ in range(config.batch_size)]

        # 裁剪 span，确保不越界
        spans = []
        for s in starts:
            end = min(s + span_len + 1, doc_len)
            span = doc_ids[0, s:end]
            if span.shape[0] < 2:
                continue
            spans.append(span)

        if not spans:
            continue

        # 对齐长度（取最短）
        min_len = min(sp.shape[0] for sp in spans)
        batch = torch.stack([sp[:min_len] for sp in spans])  # [B, T]

        input_ids = batch[:, :-1]  # [B, T-1]
        targets = batch[:, 1:]     # [B, T-1]

        outputs = model(input_ids, use_cache=False)
        logits = outputs.logits    # [B, T-1, V]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    model.eval()
    # 关闭所有参数梯度
    for param in model.parameters():
        param.requires_grad = False

    return losses


# =============================================================================
# 答案生成
# =============================================================================


def build_prompt(context: str, question: str) -> str:
    """构建 text completion 格式的 prompt（适合 base 模型）。

    Args:
        context: 文档上下文
        question: 问题

    Returns:
        格式化后的 prompt 字符串
    """
    return f"{context}\n\nQuestion: {question}\nAnswer:"


@torch.no_grad()
def generate_answer(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 32,
    max_ctx: int = 32768,
) -> str:
    """生成问题答案。

    Args:
        model: HuggingFace 模型
        tokenizer: 分词器
        prompt: 完整 prompt 字符串
        max_new_tokens: 最多生成的 token 数
        max_ctx: 推理时最大 context 长度（超出截断）

    Returns:
        生成的答案字符串（第一行，strip）
    """
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_ctx,
    )
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)

    output_ids = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )

    # 只 decode 生成部分
    generated_ids = output_ids[0, input_ids.shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    answer = answer.strip().split("\n")[0].strip()
    return answer


# =============================================================================
# 答案归一化
# =============================================================================


def normalize_answer(s: str) -> str:
    """归一化答案用于比较：lowercase + 去标点 + strip。"""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = " ".join(s.split())
    return s


# =============================================================================
# BABILong 数据加载
# =============================================================================


def load_babilong_task(
    task: str,
    context_length: str = "0k",
    split: str = "test",
) -> list[dict[str, Any]]:
    """加载 BABILong 任务数据。

    Args:
        task: 任务名称（qa1-qa10）
        context_length: 上下文长度变体
        split: 数据集分割

    Returns:
        包含 'input', 'question', 'answer' 字段的样本列表
    """
    if not HAS_DATASETS:
        raise ImportError("需要 datasets 库：pip install datasets")

    task_name = BABILONG_TASKS.get(task)
    if not task_name:
        raise ValueError(
            f"未知任务: {task}，可选: {list(BABILONG_TASKS.keys())}"
        )

    logger.info(f"加载 BABILong {task} ({task_name}) @ {context_length}")

    dataset = load_dataset(
        "RMT-team/babilong",
        f"{task_name}_{context_length}",
        split=split,
    )

    examples = []
    for item in dataset:
        examples.append(
            {
                "input": item.get("input", item.get("context", "")),
                "question": item.get("question", ""),
                "answer": item.get("answer", item.get("target", "")),
            }
        )

    return examples


# =============================================================================
# 评测函数
# =============================================================================


def evaluate(
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    device: torch.device,
    qttt_cfg: QTTTConfig | None = None,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    """在 BABILong 样本上评测模型。

    Args:
        model: HuggingFace 模型
        tokenizer: 分词器
        examples: 样本列表
        device: 设备
        qttt_cfg: 非 None 时开启 qTTT 更新；None 时为纯 baseline
        max_new_tokens: 生成时最大新 token 数

    Returns:
        包含 accuracy / correct / total / results / loss_curves 的字典
    """
    correct = 0
    total = 0
    results = []
    loss_curves = []

    max_ctx = qttt_cfg.max_ctx if qttt_cfg else 32768
    mode = "qTTT" if qttt_cfg else "Baseline"

    for example in tqdm(examples, desc=mode):
        context = example["input"]
        question = example["question"]
        gold = example["answer"]

        prompt = build_prompt(context, question)

        # 将文档 tokenize（仅 context 部分，不含问题）
        doc_losses: list[float] = []

        if qttt_cfg is not None:
            # 保存 W_Q
            saved = save_q_weights(model)

            # Tokenize 文档部分（不含问题）用于 qTTT 训练信号
            doc_encoded = tokenizer(
                context,
                return_tensors="pt",
                truncation=True,
                max_length=qttt_cfg.max_ctx,
            )
            doc_ids = doc_encoded["input_ids"].to(device)

            # qTTT 梯度更新
            doc_losses = qttt_update(model, doc_ids, qttt_cfg)

        # 生成答案
        predicted = generate_answer(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            max_ctx=max_ctx,
        )

        if qttt_cfg is not None:
            # 恢复 W_Q
            restore_q_weights(model, saved)

        # 比较答案
        pred_norm = normalize_answer(predicted)
        gold_norm = normalize_answer(gold)
        is_correct = gold_norm in pred_norm or pred_norm in gold_norm

        if is_correct:
            correct += 1
        total += 1

        if doc_losses:
            loss_curves.append(doc_losses)

        results.append(
            {
                "question": question,
                "predicted": predicted,
                "gold": gold,
                "correct": is_correct,
            }
        )

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
        "loss_curves": loss_curves,
    }


# =============================================================================
# 主程序
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="qTTT × Qwen3-7B × BABILong 评测"
    )

    # 模型参数
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-7B",
        help="HuggingFace 模型名称",
    )

    # 任务参数
    parser.add_argument(
        "--task",
        type=str,
        default="qa1",
        choices=list(BABILONG_TASKS.keys()),
        help="BABILong 任务（qa1-qa10）",
    )
    parser.add_argument(
        "--context-length",
        type=str,
        default="16k",
        choices=CONTEXT_LENGTHS,
        help="上下文长度变体",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="最多评测的样本数",
    )

    # qTTT 参数
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="qTTT 梯度步数",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="qTTT 学习率",
    )
    parser.add_argument(
        "--span-len",
        type=int,
        default=512,
        help="qTTT 训练 span 长度",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="qTTT 训练 batch size",
    )
    parser.add_argument(
        "--max-ctx",
        type=int,
        default=32768,
        help="推理最大 context 长度（超出截断）",
    )

    # 运行模式
    parser.add_argument(
        "--compare",
        action="store_true",
        default=True,
        help="同时跑 baseline + qTTT 并打印对比表（默认开启）",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="只跑 baseline，跳过 qTTT",
    )

    # 设备和输出
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备（cuda / cpu / mps）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="保存 JSON 结果的路径",
    )

    args = parser.parse_args()

    # 选择设备
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，退回 CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"使用设备: {device}")

    # 加载模型
    model, tokenizer = load_model(args.model, device)

    # 加载数据
    logger.info(f"加载任务: {args.task} @ {args.context_length}")
    examples = load_babilong_task(args.task, args.context_length)
    if args.max_examples:
        examples = examples[: args.max_examples]
    logger.info(f"共 {len(examples)} 个样本")

    # qTTT 配置
    qttt_cfg = QTTTConfig(
        num_steps=args.num_steps,
        lr=args.lr,
        span_len=args.span_len,
        batch_size=args.batch_size,
        max_ctx=args.max_ctx,
    )

    # 打印表头
    print()
    print("=== qTTT BABILong 评测 ===")
    print(f"模型: {args.model}")
    print(f"任务: {args.task} @ {args.context_length}")
    print(f"样本数: {len(examples)}")
    print()

    baseline_results = None
    qttt_results = None

    # Baseline
    logger.info("运行 Baseline...")
    baseline_results = evaluate(
        model, tokenizer, examples, device,
        qttt_cfg=None,
    )
    baseline_acc = baseline_results["accuracy"] * 100
    print(
        f"Baseline:  {baseline_acc:.1f}%"
        f" ({baseline_results['correct']}/{baseline_results['total']})"
    )

    # qTTT
    if not args.baseline_only:
        logger.info("运行 qTTT...")
        qttt_results = evaluate(
            model, tokenizer, examples, device,
            qttt_cfg=qttt_cfg,
        )
        qttt_acc = qttt_results["accuracy"] * 100
        delta = qttt_acc - baseline_acc

        # 汇总 loss 曲线
        all_losses = qttt_results["loss_curves"]
        if all_losses:
            first_losses = [curve[0] for curve in all_losses if curve]
            last_losses = [curve[-1] for curve in all_losses if curve]
            avg_first = sum(first_losses) / len(first_losses)
            avg_last = sum(last_losses) / len(last_losses)
            loss_str = f"  [loss: {avg_first:.2f} → {avg_last:.2f}, 步数={args.num_steps}]"
        else:
            loss_str = ""

        sign = "+" if delta >= 0 else ""
        print(
            f"qTTT:      {qttt_acc:.1f}%"
            f" ({qttt_results['correct']}/{qttt_results['total']})"
            f"  [{sign}{delta:.1f}pp]"
        )
        if loss_str:
            print(f"qTTT loss:{loss_str}")

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data: dict[str, Any] = {
            "model": args.model,
            "task": args.task,
            "context_length": args.context_length,
            "num_examples": len(examples),
            "qttt_config": {
                "num_steps": args.num_steps,
                "lr": args.lr,
                "span_len": args.span_len,
                "batch_size": args.batch_size,
                "max_ctx": args.max_ctx,
            },
        }

        if baseline_results:
            output_data["baseline"] = {
                "accuracy": baseline_results["accuracy"],
                "correct": baseline_results["correct"],
                "total": baseline_results["total"],
                "per_sample": baseline_results["results"],
            }

        if qttt_results:
            all_losses = qttt_results["loss_curves"]
            output_data["qttt"] = {
                "accuracy": qttt_results["accuracy"],
                "correct": qttt_results["correct"],
                "total": qttt_results["total"],
                "per_sample": qttt_results["results"],
                "avg_loss_curves": (
                    {
                        "first": sum(c[0] for c in all_losses if c) / len(all_losses),
                        "last": sum(c[-1] for c in all_losses if c) / len(all_losses),
                    }
                    if all_losses
                    else {}
                ),
            }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存: {output_path}")

    print()


if __name__ == "__main__":
    main()
