#!/usr/bin/env python3
"""从 locomo10.json 生成英文 Stage 4 训练数据。

格式与 Stage 1-3 完全一致（只换英文内容）：
    【写入】turn1
    【写入】turn2
    ...
    【查询】question
    【记忆】answer

划分：
    conv 0-7 → 训练集 (oracle_en.jsonl)
    conv 8-9 → 评测集 (oracle_en_eval.jsonl，保留给 eval_locomo10.py)

用法：
    python scripts/build_locomo_train.py \\
        --data data/locomo10.json \\
        --out-train data/oracle_en.jsonl \\
        --out-eval data/oracle_en_eval.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path


TRAIN_CONV_RANGE = (0, 8)   # [0, 8)
EVAL_CONV_RANGE  = (8, 10)  # [8, 10)

# 跳过 adversarial（cat=5，答案是 "I don't know"），无法学
SKIP_CATEGORIES = {5}


def extract_turns(conv_dict: dict) -> list[str]:
    turns = []
    for i in range(1, 100):
        key = f"session_{i}"
        if key not in conv_dict:
            break
        session = conv_dict[key]
        if not isinstance(session, list):
            continue
        for turn in session:
            if isinstance(turn, dict):
                speaker = turn.get("speaker", "")
                text = turn.get("text", "").strip()
                if text:
                    turns.append(f"{speaker}: {text}" if speaker else text)
            elif isinstance(turn, str) and turn.strip():
                turns.append(turn.strip())
    return turns


def build_samples(data: list[dict], conv_range: tuple[int, int],
                  max_turns_per_sample: int) -> list[dict]:
    """为每个 QA pair 生成一条训练样本。

    每条样本的 history 是该对话的全量 turns（分段写入由训练脚本的 commit 机制处理）。
    这里只截取最近 max_turns_per_sample 条，保证 tokenize 后不超 2048 tokens。
    """
    samples = []
    for conv_idx in range(*conv_range):
        item = data[conv_idx]
        conv_dict = item["conversation"]
        qa_list = item.get("qa", [])
        turns = extract_turns(conv_dict)

        # 所有 QA 共用同一段 history（全部轮次）
        # 截取最近 N 条以控制序列长度（约 15 tokens/turn → 100 turns ≈ 1500 tokens）
        history = turns[-max_turns_per_sample:] if len(turns) > max_turns_per_sample else turns

        for qa in qa_list:
            cat = qa.get("category", 0)
            if cat in SKIP_CATEGORIES:
                continue
            question = qa.get("question", "").strip()
            answer = str(qa.get("answer", "")).strip()
            if not question or not answer:
                continue

            samples.append({
                "conv_idx": conv_idx,
                "category": cat,
                "history": [{"role": "user", "content": t} for t in history],
                "query": question,
                "target_memory": answer,
            })

    return samples


def write_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {len(samples)} samples → {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/locomo10.json")
    p.add_argument("--out-train", default="data/oracle_en.jsonl")
    p.add_argument("--out-eval", default="data/oracle_en_eval.jsonl")
    p.add_argument("--max-turns", type=int, default=100,
                   help="Max turns per sample (controls context length; ~15 tok/turn)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} conversations")

    train_samples = build_samples(data, TRAIN_CONV_RANGE, args.max_turns)
    eval_samples  = build_samples(data, EVAL_CONV_RANGE,  args.max_turns)

    # 训练集打乱
    random.shuffle(train_samples)

    write_jsonl(train_samples, Path(args.out_train))
    write_jsonl(eval_samples,  Path(args.out_eval))

    # 分类统计
    from collections import Counter
    cat_names = {1:"single-hop", 2:"temporal", 3:"multi-hop", 4:"open-ended"}
    train_cats = Counter(s["category"] for s in train_samples)
    print("Train category dist:", {cat_names.get(k, k): v for k, v in sorted(train_cats.items())})
    eval_cats = Counter(s["category"] for s in eval_samples)
    print("Eval  category dist:", {cat_names.get(k, k): v for k, v in sorted(eval_cats.items())})

    # 打印几条样本看格式
    print("\n=== Sample ===")
    s = train_samples[0]
    print("history turns:", len(s["history"]))
    print("query:", s["query"])
    print("target:", s["target_memory"])
    print("history[-2]:", s["history"][-2]["content"][:100])


if __name__ == "__main__":
    main()
