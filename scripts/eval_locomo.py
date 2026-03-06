#!/usr/bin/env python3
# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
LOCOMO 评测脚本（LLM-as-Judge 评分，对齐 mem0 论文指标）。

评测对象：Memory Oracle（Qwen3.5-0.8B + NLM）
评测数据：LOCOMO 数据集（snap-research/locomo）
评测方式：LLM-as-Judge（gpt-4o-mini），输出 0-10 分，取平均

mem0 baseline：66.9%（来自论文）

输出：
  - JSON 结果文件（每题得分 + 汇总）
  - 控制台汇总（single-hop / multi-hop / temporal 三类）

用法：
    # 用 MemoryOracle 评测
    uv run python scripts/eval_locomo.py \\
        --oracle checkpoints/oracle_stage3/ \\
        --data data/locomo10.json \\
        --judge gpt-4o-mini \\
        --output results/locomo_oracle.json

    # 无记忆 baseline（直接用 Qwen3.5，不注入 NLM）
    uv run python scripts/eval_locomo.py \\
        --baseline-only \\
        --base-model Qwen/Qwen3.5-0.8B \\
        --data data/locomo10.json \\
        --output results/locomo_baseline.json

    # 快速验证（前 50 题）
    uv run python scripts/eval_locomo.py \\
        --oracle checkpoints/oracle_stage3/ \\
        --data data/locomo10.json \\
        --max-examples 50 --output results/quick_test.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from titans.qwen35_injection import freeze_memory_updates, unfreeze_memory_updates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM-as-Judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """你是一个公正的评估专家。根据给定的问题、参考答案和模型预测，
评估模型预测的质量。打分范围 0-10（10分=完全正确，0分=完全错误）。
只输出一个整数分数，不要解释。"""

JUDGE_USER_TEMPLATE = """问题：{question}

参考答案：{reference}

模型预测：{prediction}

请给模型预测打分（0-10分）："""


class LLMJudge:
    """LLM-as-Judge 评分器。"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        max_retries: int = 3,
    ) -> None:
        try:
            import openai
            self.client = openai.OpenAI(base_url=base_url) if base_url else openai.OpenAI()
        except ImportError:
            raise ImportError("openai required: pip install openai")
        self.model = model
        self.max_retries = max_retries

    def score(self, question: str, reference: str, prediction: str) -> float:
        """返回 0-10 分，标准化到 0-1。"""
        user_prompt = JUDGE_USER_TEMPLATE.format(
            question=question,
            reference=reference,
            prediction=prediction,
        )
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=10,
                    temperature=0.0,
                )
                raw = resp.choices[0].message.content.strip()
                # 提取第一个数字
                import re
                m = re.search(r"\d+", raw)
                if m:
                    score = min(10, max(0, int(m.group(0))))
                    return score / 10.0
            except Exception as e:
                log.warning("Judge error (attempt %d): %s", attempt + 1, e)
                time.sleep(2 ** attempt)
        return 0.0


# ---------------------------------------------------------------------------
# 简单 F1 评分（作为 LLM-judge 的 fallback）
# ---------------------------------------------------------------------------


def f1_score(prediction: str, reference: str) -> float:
    """Token-level F1（简单空格分词）。"""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# LOCOMO 数据加载
# ---------------------------------------------------------------------------


def load_locomo(data_path: str | Path) -> list[dict]:
    """加载 LOCOMO 数据集。

    支持两种格式：
    1. 本地 JSON 文件（snap-research/locomo 格式）
    2. 通过 HuggingFace datasets 加载

    LOCOMO 字段：
        conversation: list of session turns
        question:     问题字符串
        answer:       参考答案
        category:     single-hop / multi-hop / temporal / open-ended
    """
    data_path = Path(data_path)

    if data_path.exists():
        log.info("Loading LOCOMO from local file: %s", data_path)
        with data_path.open(encoding="utf-8") as f:
            data = json.load(f)
        # 支持 list 或 dict（某些版本是 {"data": [...]}）
        if isinstance(data, dict):
            for key in ("data", "samples", "qa_pairs"):
                if key in data:
                    data = data[key]
                    break
            else:
                known_keys = list(data.keys())[:5]
                raise ValueError(
                    f"Unknown LOCOMO dict format. Top-level keys: {known_keys}. "
                    "Expected 'data', 'samples', or 'qa_pairs'."
                )
        return data

    # 尝试从 HuggingFace 加载
    log.info("Local file not found, trying HuggingFace: snap-research/locomo")
    try:
        from datasets import load_dataset
        ds = load_dataset("snap-research/locomo", split="test", trust_remote_code=True)
        return [dict(item) for item in ds]
    except Exception as e:
        raise FileNotFoundError(
            f"LOCOMO data not found at {data_path} and HuggingFace load failed: {e}"
        )


def flatten_qa_pairs(locomo_data: list[dict]) -> list[dict]:
    """将 LOCOMO 数据展平为 QA pair 列表。

    每个 QA pair 包含 conv_idx（该对话在 locomo_data 中的位置），
    用于在 eval_with_oracle 中正确分组——避免 id() 在 JSON 解析对象上失效的问题。
    """
    qa_pairs = []
    for conv_idx, item in enumerate(locomo_data):
        # 提取对话历史
        conv = item.get("conversation", item.get("dialog", item.get("history", [])))
        if isinstance(conv, str):
            conv_list = [{"text": conv}]
        elif isinstance(conv, list):
            conv_list = conv
        else:
            conv_list = []

        # 提取 QA pairs（支持多个 question per conversation）
        questions = item.get("questions", item.get("qa_pairs", []))
        if not questions:
            q = item.get("question", "")
            a = item.get("answer", item.get("reference", ""))
            cat = item.get("category", item.get("type", "unknown"))
            if q and a:
                qa_pairs.append({
                    "conv_idx": conv_idx,
                    "conversation": conv_list,
                    "question": q,
                    "answer": a,
                    "category": cat,
                })
        else:
            for qa in questions:
                q = qa.get("question", "")
                a = qa.get("answer", qa.get("reference", ""))
                cat = qa.get("category", qa.get("type", "unknown"))
                if q and a:
                    qa_pairs.append({
                        "conv_idx": conv_idx,
                        "conversation": conv_list,
                        "question": q,
                        "answer": a,
                        "category": cat,
                    })

    return qa_pairs


# ---------------------------------------------------------------------------
# MemoryOracle 评测
# ---------------------------------------------------------------------------


def eval_with_oracle(
    oracle_ckpt: str,
    qa_pairs: list[dict],
    base_model: str,
    device: str,
    max_examples: int | None,
    judge: LLMJudge | None,
) -> list[dict]:
    """用 MemoryOracle 对每个 QA pair 评测。

    对于每个 conversation：
    1. 把所有 turns 逐条 write() 进入 oracle
    2. 用 question 触发 oracle.read() 获得记忆摘要
    3. 用记忆摘要 + question 生成最终答案
    4. 用 judge / F1 评分
    """
    from titans.memory_oracle import MemoryOracle

    log.info("Loading MemoryOracle from %s", oracle_ckpt)
    oracle = MemoryOracle.from_pretrained(
        oracle_ckpt,
        base_model=base_model,
        device=device,
    )

    # oracle.model 已加载，复用它生成答案
    gen_model = oracle.model
    gen_tokenizer = oracle.tokenizer

    if max_examples:
        qa_pairs = qa_pairs[:max_examples]

    # 按 conv_idx 分组（flatten_qa_pairs 保证同一对话有相同的 conv_idx）
    conv_groups = _group_by_conv_idx(qa_pairs)
    n_convs = len(conv_groups)
    log.info("Evaluating %d QA pairs across %d conversations", len(qa_pairs), n_convs)

    results = []
    for group in tqdm(conv_groups, desc="Oracle eval", total=n_convs):
        oracle.reset()
        conv = group[0]["conversation"]

        # Write 对话历史（同一 conversation 只 write 一次，所有问题共用）
        for turn in conv:
            text = _extract_conv_text(turn)
            if text:
                oracle.write(text)

        # 对该 conversation 的每个问题单独 read（NLM frozen，不写入）
        for qa in group:
            question = qa["question"]
            reference = qa["answer"]
            category = qa["category"]

            memory_summary = oracle.read(question)

            answer_prompt = (
                f"背景记忆：{memory_summary}\n\n"
                f"请根据以上记忆回答问题（简洁准确，不超过50字）：\n"
                f"问题：{question}\n答案："
            )
            # oracle.read() 已经调用了 unfreeze，_generate_answer 在写入模式下运行；
            # 为了不让答案生成污染 NLM，在 generate 前后显式 freeze/unfreeze。
            freeze_memory_updates(gen_model)
            try:
                prediction = _generate_answer(gen_model, gen_tokenizer, answer_prompt, device)
            finally:
                unfreeze_memory_updates(gen_model)

            score = judge.score(question, reference, prediction) if judge else f1_score(prediction, reference)

            results.append({
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "memory_summary": memory_summary,
                "category": category,
                "score": score,
            })

    return results


def _group_by_conv_idx(qa_pairs: list[dict]) -> list[list[dict]]:
    """按 conv_idx 分组，保持对话顺序。返回 list of groups（每 group 是同一对话的所有 QA）。

    使用 flatten_qa_pairs 注入的 conv_idx（基于位置，稳定）而非 id()（基于内存地址，不稳定）。
    """
    groups: dict[int, list[dict]] = {}
    for qa in qa_pairs:
        cid = qa["conv_idx"]
        if cid not in groups:
            groups[cid] = []
        groups[cid].append(qa)
    # 按 conv_idx 排序，保证顺序确定
    return [groups[k] for k in sorted(groups.keys())]


def _extract_conv_text(turn: Any) -> str:
    if isinstance(turn, str):
        return turn.strip()
    if isinstance(turn, dict):
        for k in ("text", "content", "utterance", "message"):
            if k in turn:
                speaker = turn.get("speaker", turn.get("role", ""))
                text = str(turn[k]).strip()
                return f"{speaker}：{text}" if speaker else text
    return ""


@torch.no_grad()
def _generate_answer(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str,
    max_new_tokens: int = 100,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# 无记忆 baseline
# ---------------------------------------------------------------------------


def eval_baseline(
    base_model: str,
    qa_pairs: list[dict],
    device: str,
    max_examples: int | None,
    judge: LLMJudge | None,
) -> list[dict]:
    """直接用基模型（无 NLM）回答问题，作为 no-memory baseline。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading baseline model: %s", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if max_examples:
        qa_pairs = qa_pairs[:max_examples]

    results = []
    for qa in tqdm(qa_pairs, desc="Baseline"):
        conv = qa["conversation"]
        question = qa["question"]
        reference = qa["answer"]
        category = qa["category"]

        # 取最近 5 条对话作为 context（sliding window baseline）
        recent = conv[-10:] if len(conv) > 10 else conv
        context = "\n".join(_extract_conv_text(t) for t in recent if _extract_conv_text(t))

        prompt = (
            f"对话历史（节选）：\n{context}\n\n"
            f"请根据以上对话回答（简洁，不超过50字）：\n"
            f"问题：{question}\n答案："
        )
        prediction = _generate_answer(model, tokenizer, prompt, device)

        if judge is not None:
            score = judge.score(question, reference, prediction)
        else:
            score = f1_score(prediction, reference)

        results.append({
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "category": category,
            "score": score,
        })

    return results


# ---------------------------------------------------------------------------
# 汇总统计
# ---------------------------------------------------------------------------


def summarize(results: list[dict]) -> dict:
    """计算总体 + 分类别的平均分。"""
    from collections import defaultdict

    cat_scores: dict[str, list[float]] = defaultdict(list)
    all_scores = []
    for r in results:
        s = r["score"]
        all_scores.append(s)
        cat_scores[r.get("category", "unknown")].append(s)

    summary = {
        "total": len(all_scores),
        "overall": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "by_category": {
            cat: {
                "n": len(scores),
                "avg": sum(scores) / len(scores),
            }
            for cat, scores in sorted(cat_scores.items())
        },
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Memory Oracle on LOCOMO")

    p.add_argument("--oracle", default=None,
                   help="MemoryOracle checkpoint directory")
    p.add_argument("--baseline-only", action="store_true",
                   help="Only run no-memory baseline (no NLM)")
    p.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B",
                   help="Base model for oracle or baseline")
    p.add_argument("--data", required=True,
                   help="LOCOMO data: local JSON path or HuggingFace dataset")
    p.add_argument("--output", required=True, help="Output JSON file for results")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Limit number of QA pairs (for quick testing)")
    p.add_argument("--judge", default=None,
                   help="Judge model name (e.g. gpt-4o-mini); if None, use F1 fallback")
    p.add_argument("--judge-base-url", default=None,
                   help="Custom base URL for judge OpenAI client")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 加载 LOCOMO
    raw_data = load_locomo(args.data)
    qa_pairs = flatten_qa_pairs(raw_data)
    log.info("LOCOMO: %d QA pairs total", len(qa_pairs))

    # 评分器
    judge: LLMJudge | None = None
    if args.judge:
        judge = LLMJudge(model=args.judge, base_url=args.judge_base_url)
        log.info("Using LLM-as-Judge: %s", args.judge)
    else:
        log.info("No judge specified — using F1 fallback")

    # 评测
    if args.baseline_only:
        results = eval_baseline(args.base_model, qa_pairs, args.device, args.max_examples, judge)
        mode = "baseline"
    else:
        if args.oracle is None:
            log.error("Either --oracle or --baseline-only is required")
            sys.exit(1)
        results = eval_with_oracle(
            args.oracle, qa_pairs, args.base_model, args.device, args.max_examples, judge
        )
        mode = "oracle"

    # 汇总
    summary = summarize(results)

    log.info("=== %s Results ===", mode.upper())
    log.info("Total QA pairs: %d", summary["total"])
    log.info("Overall score:  %.1f%%", summary["overall"] * 100)
    for cat, stat in summary["by_category"].items():
        log.info("  %-20s  n=%4d  avg=%.1f%%", cat, stat["n"], stat["avg"] * 100)
    log.info("mem0 baseline: 66.9%%")

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"mode": mode, "summary": summary, "results": results},
                  f, ensure_ascii=False, indent=2)
    log.info("Results saved → %s", output_path)


if __name__ == "__main__":
    main()
