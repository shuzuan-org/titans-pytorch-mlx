#!/usr/bin/env python3
# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
构造 Memory Oracle 训练数据（Write→Read 格式）。

支持以下数据来源：
  Stage 1：CMRC 2018、DuReader（文档理解）
  Stage 2：MSC（Multi-Session Chat，多 session 对话记忆）

输出格式（JSONL，每行一个样本）：
{
  "history": [{"role": "write", "content": "..."},...],
  "query":   "...",
  "target_memory": "..."
}

用法：
    # Stage 1：文档理解
    uv run python scripts/build_oracle_data.py \\
        --stage 1 --output data/oracle_stage1.jsonl

    # Stage 2：对话记忆（MSC）
    uv run python scripts/build_oracle_data.py \\
        --stage 2 --output data/oracle_stage2.jsonl

    # 合并两者（Stage 1 + 2 可同目录，名字不同）
    cat data/oracle_stage1.jsonl data/oracle_stage2.jsonl > data/oracle_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    log.error("datasets library not found. Run: pip install datasets")


# ---------------------------------------------------------------------------
# Stage 1：文档理解（CMRC / DuReader → Write→Read）
# ---------------------------------------------------------------------------


def _cmrc_samples(split: str = "train") -> Generator[dict, None, None]:
    """CMRC 2018：passage → Write；question → Read；answer → target。

    数据集字段：
        context, question, answers (list of strings)
    """
    log.info("Loading CMRC 2018 (%s)", split)
    ds = load_dataset("hfl/cmrc2018", split=split, trust_remote_code=True)

    for item in ds:
        context: str = item["context"]
        question: str = item["question"]
        answers: list[str] = item["answers"]["text"]
        if not answers:
            continue

        # 取第一个答案作为 target（CMRC 通常多个答案相同）
        target = answers[0].strip()
        if not target:
            continue

        yield {
            "history": [{"role": "write", "content": context.strip()}],
            "query": question.strip(),
            "target_memory": target,
            "source": "cmrc2018",
        }


def _dureader_samples(split: str = "train", max_samples: int = 30000) -> Generator[dict, None, None]:
    """DuReader（简化版）：passage → Write；question → Read；answer → target。

    HuggingFace 上有多个 DuReader 变体，这里尝试 dureader_robust。
    若找不到则 skip。
    """
    dataset_names = ["dureader_robust", "baidu/dureader"]
    ds = None
    for name in dataset_names:
        try:
            log.info("Trying to load DuReader: %s", name)
            ds = load_dataset(name, split=split, trust_remote_code=True)
            log.info("Loaded %s", name)
            break
        except Exception as e:
            log.warning("Failed to load %s: %s", name, e)

    if ds is None:
        log.warning("DuReader not available, skipping.")
        return

    count = 0
    for item in ds:
        if count >= max_samples:
            break

        # dureader_robust 字段：question, context, answers
        context = item.get("context", item.get("passage", "")).strip()
        question = item.get("question", "").strip()
        answers_raw = item.get("answers", {})

        if isinstance(answers_raw, dict):
            answers = answers_raw.get("text", [])
        elif isinstance(answers_raw, list):
            answers = answers_raw
        else:
            answers = []

        answers = [a.strip() for a in answers if a.strip()]
        if not context or not question or not answers:
            continue

        yield {
            "history": [{"role": "write", "content": context}],
            "query": question,
            "target_memory": answers[0],
            "source": "dureader",
        }
        count += 1


# ---------------------------------------------------------------------------
# Stage 2：对话记忆（MSC → Write→Read）
# ---------------------------------------------------------------------------


def _msc_samples(split: str = "train", max_sessions_per_sample: int = 4) -> Generator[dict, None, None]:
    """MSC (Multi-Session Chat)：session 1..N-1 → Write；session N 问题 → Read。

    MSC 字段（nayohan/multi_session_chat）：
        dialog: list of session dicts，每个 session 有 "dialog" list of turns

    构造逻辑：
        - 取 session[:-1] 的对话内容作为 history（逐轮写入）
        - 取 session[-1] 第一条 user 发言作为 query
        - 取 session[-1] 第一条 assistant 发言作为 target_memory（近似）
    """
    log.info("Loading MSC (multi_session_chat) split=%s", split)
    try:
        ds = load_dataset("nayohan/multi_session_chat", split=split, trust_remote_code=True)
    except Exception as e:
        log.warning("MSC not available: %s", e)
        return

    for item in ds:
        # 字段结构因版本而异，做鲁棒处理
        sessions: list[Any] = item.get("dialog", item.get("sessions", []))
        if not isinstance(sessions, list) or len(sessions) < 2:
            continue

        # Truncate to max_sessions_per_sample + 1
        sessions = sessions[: max_sessions_per_sample + 1]

        # Write sessions: 前 N-1 session 的所有对话轮
        write_turns: list[dict] = []
        for session in sessions[:-1]:
            turns = session if isinstance(session, list) else session.get("dialog", [])
            for turn in turns:
                text = _extract_turn_text(turn)
                if text:
                    write_turns.append({"role": "write", "content": text})

        if not write_turns:
            continue

        # Read session: 最后一个 session 的第一个问题和回答
        last_session = sessions[-1]
        last_turns = last_session if isinstance(last_session, list) else last_session.get("dialog", [])
        if len(last_turns) < 2:
            continue

        query_text = _extract_turn_text(last_turns[0])
        target_text = _extract_turn_text(last_turns[1])
        if not query_text or not target_text:
            continue

        yield {
            "history": write_turns,
            "query": query_text,
            "target_memory": target_text,
            "source": "msc",
        }


def _extract_turn_text(turn: Any) -> str:
    """从不同格式的 turn 对象提取文本。"""
    if isinstance(turn, str):
        return turn.strip()
    if isinstance(turn, dict):
        for key in ("text", "content", "utterance", "message"):
            if key in turn:
                return str(turn[key]).strip()
    return ""


# ---------------------------------------------------------------------------
# Stage 2：MemGPT/MSC-Self-Instruct
# ---------------------------------------------------------------------------


def _msc_self_instruct_samples(split: str = "train") -> Generator[dict, None, None]:
    """MemGPT/MSC-Self-Instruct：直接有 summary 字段，转为 target_memory。"""
    log.info("Loading MemGPT/MSC-Self-Instruct split=%s", split)
    try:
        ds = load_dataset("MemGPT/MSC-Self-Instruct", split=split, trust_remote_code=True)
    except Exception as e:
        log.warning("MSC-Self-Instruct not available: %s", e)
        return

    for item in ds:
        history_text = item.get("history", item.get("context", "")).strip()
        query = item.get("query", item.get("question", "")).strip()
        target = item.get("summary", item.get("answer", item.get("response", ""))).strip()

        if not history_text or not query or not target:
            continue

        # Split history into write entries (by newline / sentence)
        history = [
            {"role": "write", "content": line.strip()}
            for line in history_text.split("\n")
            if line.strip()
        ]
        if not history:
            history = [{"role": "write", "content": history_text}]

        yield {
            "history": history,
            "query": query,
            "target_memory": target,
            "source": "msc_self_instruct",
        }


# ---------------------------------------------------------------------------
# Stage 3：重要性标签注入
# ---------------------------------------------------------------------------


def add_importance_labels(sample: dict) -> dict:
    """为 Stage 3 训练添加重要性标签。

    规则：target_memory 中出现过的历史条目 → importance=1，其余 → importance=0。
    正样本约占 10-20%（随 target 内容而变）。
    """
    target = sample["target_memory"].lower()
    history = sample["history"]

    labeled = []
    for turn in history:
        content = turn["content"].lower()
        # 简单词重叠：target 中有该条目的任意 5-gram
        words = content.split()
        is_important = False
        if len(words) >= 3:
            for i in range(len(words) - 2):
                ngram = " ".join(words[i : i + 3])
                if ngram in target:
                    is_important = True
                    break
        labeled.append({**turn, "importance": 1 if is_important else 0})

    return {**sample, "history": labeled}


# ---------------------------------------------------------------------------
# 写出 JSONL
# ---------------------------------------------------------------------------


def write_jsonl(
    generator: Generator[dict, None, None],
    output_path: Path,
    max_samples: int | None = None,
    add_importance: bool = False,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for sample in generator:
            if max_samples and count >= max_samples:
                break
            if add_importance:
                sample = add_importance_labels(sample)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
            if count % 1000 == 0:
                log.info("  Written %d samples", count)
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Memory Oracle training data")
    p.add_argument("--stage", type=int, choices=[1, 2, 3], default=1,
                   help="Training stage: 1=doc-understanding, 2=dialog-memory, 3=importance-aware")
    p.add_argument("--output", required=True, help="Output JSONL file path")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Max samples per source (None = all)")
    p.add_argument("--split", default="train", help="Dataset split (train/validation)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    if not HAS_DATASETS:
        sys.exit(1)

    args = parse_args()
    random.seed(args.seed)
    output_path = Path(args.output)

    if args.stage == 1:
        log.info("=== Stage 1: Document Understanding ===")
        # CMRC 2018
        cmrc_path = output_path.with_name(output_path.stem + "_cmrc.jsonl")
        n = write_jsonl(_cmrc_samples(args.split), cmrc_path, args.max_samples)
        log.info("CMRC: %d samples → %s", n, cmrc_path)

        # DuReader
        dr_path = output_path.with_name(output_path.stem + "_dureader.jsonl")
        n = write_jsonl(_dureader_samples(args.split, args.max_samples or 30000),
                        dr_path, args.max_samples)
        log.info("DuReader: %d samples → %s", n, dr_path)

        # 合并
        log.info("Merging to %s", output_path)
        with output_path.open("w", encoding="utf-8") as out:
            for sub in [cmrc_path, dr_path]:
                if sub.exists():
                    with sub.open() as f:
                        for line in f:
                            out.write(line)
        log.info("Stage 1 done → %s", output_path)

    elif args.stage == 2:
        log.info("=== Stage 2: Dialog Memory ===")

        # MSC
        msc_path = output_path.with_name(output_path.stem + "_msc.jsonl")
        n = write_jsonl(_msc_samples(args.split), msc_path, args.max_samples)
        log.info("MSC: %d samples → %s", n, msc_path)

        # MemGPT/MSC-Self-Instruct
        instruct_path = output_path.with_name(output_path.stem + "_instruct.jsonl")
        n = write_jsonl(_msc_self_instruct_samples(args.split), instruct_path, args.max_samples)
        log.info("MSC-Self-Instruct: %d samples → %s", n, instruct_path)

        # 合并
        with output_path.open("w", encoding="utf-8") as out:
            for sub in [msc_path, instruct_path]:
                if sub.exists():
                    with sub.open() as f:
                        for line in f:
                            out.write(line)
        log.info("Stage 2 done → %s", output_path)

    elif args.stage == 3:
        log.info("=== Stage 3: Importance-Aware (expects Stage 2 data as input) ===")
        # 读入 Stage 2 数据，添加重要性标签
        # 用法：--output data/oracle_stage3.jsonl（会从同级目录找 stage2 数据）
        stage2_path = output_path.with_name(output_path.stem.replace("stage3", "stage2") + ".jsonl")
        if not stage2_path.exists():
            log.error("Stage 2 data not found at %s. Build stage 2 first.", stage2_path)
            sys.exit(1)

        log.info("Adding importance labels from %s", stage2_path)
        count = 0
        with output_path.open("w", encoding="utf-8") as out:
            with stage2_path.open() as f:
                for line in f:
                    sample = json.loads(line)
                    labeled = add_importance_labels(sample)
                    out.write(json.dumps(labeled, ensure_ascii=False) + "\n")
                    count += 1
        log.info("Stage 3 done: %d samples → %s", count, output_path)


if __name__ == "__main__":
    main()
