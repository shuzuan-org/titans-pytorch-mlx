#!/usr/bin/env python3
"""LoCoMo-10 评测 — Memory Oracle vs No-Memory Baseline

数据格式（locomo10.json）：
    list of 10 conversations, each:
        conversation: {speaker_a, speaker_b, session_1: [...], session_2: [...], ...}
        qa: [{question, answer, evidence, category}, ...]

category 映射（LoCoMo 论文）：
    1 = single-hop    2 = temporal    3 = multi-hop
    4 = open-ended    5 = adversarial (answer is "I don't know")

用法（F1 fallback，无需 OpenAI key）：
    python scripts/eval_locomo10.py --data data/locomo10.json \\
        --oracle checkpoints/oracle_08b/stage3 \\
        --device cuda:7 --max-qa 20

用法（全量 + LLM judge）：
    python scripts/eval_locomo10.py --data data/locomo10.json \\
        --oracle checkpoints/oracle_08b/stage3 \\
        --device cuda:7 --judge gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

CATEGORY_NAMES = {1: "single-hop", 2: "temporal", 3: "multi-hop",
                  4: "open-ended", 5: "adversarial"}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_locomo10(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_turns(conv_dict: dict) -> list[str]:
    """Flatten session_1, session_2, ... → ordered list of 'Speaker: text' strings."""
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
            elif isinstance(turn, str):
                turns.append(turn.strip())
    return turns


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 (whitespace tokenization)."""
    pred_tok = set(pred.lower().split())
    gold_tok = set(str(gold).lower().split())
    if not pred_tok or not gold_tok:
        return float(pred.strip().lower() == str(gold).strip().lower())
    common = pred_tok & gold_tok
    if not common:
        return 0.0
    p = len(common) / len(pred_tok)
    r = len(common) / len(gold_tok)
    return 2 * p * r / (p + r)


class LLMJudge:
    def __init__(self, model: str, base_url: str | None = None) -> None:
        import openai
        self.client = openai.OpenAI(base_url=base_url) if base_url else openai.OpenAI()
        self.model = model

    def score(self, question: str, reference: str, prediction: str) -> float:
        prompt = (
            f"Question: {question}\n"
            f"Reference answer: {reference}\n"
            f"Model prediction: {prediction}\n\n"
            "Rate the prediction's correctness from 0 to 10 (10=fully correct). "
            "Output only the integer."
        )
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0.0,
                )
                raw = resp.choices[0].message.content.strip()
                m = re.search(r"\d+", raw)
                if m:
                    return min(10, max(0, int(m.group()))) / 10.0
            except Exception as e:
                log.warning("Judge error (attempt %d): %s", attempt + 1, e)
                time.sleep(2 ** attempt)
        return 0.0


# ---------------------------------------------------------------------------
# Oracle evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_answer(model: Any, tokenizer: Any, prompt: str, device: str,
                    max_new_tokens: int = 80) -> str:
    from titans.qwen35_injection import freeze_memory_updates, unfreeze_memory_updates
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)
    freeze_memory_updates(model)
    try:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    finally:
        unfreeze_memory_updates(model)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # 截断换行后的废话
    for stop in ("\n\n", "\nQuestion", "\nReference", "Answer:"):
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx].strip()
    return text


def eval_oracle(data: list[dict], args: argparse.Namespace,
                judge: LLMJudge | None) -> list[dict]:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from titans.memory_oracle import MemoryOracle

    BASE_MODEL = args.base_model
    log.info("Loading MemoryOracle from %s", args.oracle)
    oracle = MemoryOracle.from_pretrained(
        args.oracle,
        base_model=BASE_MODEL,
        device=args.device,
        num_memory_layers=1,
        memory_lr=0.1,
        memory_momentum=0.9,
        memory_decay=0.01,
        max_read_new_tokens=150,
    )
    # 每段 write 序列与训练格式对齐（2048 token 窗口），超出后 commit
    oracle.max_seq_len = 2048
    oracle.tokenizer.truncation_side = "left"

    results = []
    qa_count = 0

    for conv_idx, item in enumerate(data):
        conv_dict = item["conversation"]
        qa_list = item.get("qa", [])
        if not qa_list:
            continue

        turns = extract_turns(conv_dict)
        log.info("Conv %d/%d: %d turns, %d questions",
                 conv_idx + 1, len(data), len(turns), len(qa_list))

        # 分段写入：每 commit_every 轮 commit() 一次，保持与训练上下文长度一致
        oracle.reset()
        commit_every = args.commit_every
        for i, turn_text in enumerate(turns):
            oracle.write(turn_text)
            if commit_every and (i + 1) % commit_every == 0:
                oracle.commit()
                log.debug("Committed at turn %d", i + 1)

        for qa in qa_list:
            if args.max_qa and qa_count >= args.max_qa:
                return results

            question = qa["question"]
            reference = str(qa["answer"])
            cat_id = qa.get("category", 0)
            cat_name = CATEGORY_NAMES.get(cat_id, f"cat{cat_id}")

            # Skip adversarial for now (answer is "I don't know")
            if cat_id == 5 and not args.include_adversarial:
                continue

            # Memory recall
            memory_summary = oracle.read(question)

            # Answer generation
            prompt = (
                f"Background memory: {memory_summary}\n\n"
                f"Answer the question briefly (under 30 words):\n"
                f"Q: {question}\nA:"
            )
            prediction = generate_answer(
                oracle.model, oracle.tokenizer, prompt, args.device
            )

            score = judge.score(question, reference, prediction) if judge else token_f1(prediction, reference)

            results.append({
                "conv_idx": conv_idx,
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "memory_summary": memory_summary,
                "category": cat_name,
                "score": score,
            })
            qa_count += 1

            if qa_count % 20 == 1 or qa_count == 1:
                avg = sum(r["score"] for r in results) / len(results)
                log.info("[%3d] %s avg=%.3f  Q: %s",
                         qa_count, cat_name, avg, question[:60])
                log.info("  mem : %s", memory_summary[:80])
                log.info("  pred: %s", prediction[:80])
                log.info("  gold: %s", reference[:80])

    return results


def eval_baseline(data: list[dict], args: argparse.Namespace,
                  judge: LLMJudge | None) -> list[dict]:
    """No-memory baseline: last-10-turns sliding window."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from titans.qwen35_injection import freeze_memory_updates, unfreeze_memory_updates

    log.info("Loading baseline model %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        trust_remote_code=True,
    )
    model.eval()

    # Monkey-patch freeze/unfreeze as no-ops for generate_answer compatibility
    import types
    freeze_memory_updates = lambda m: None
    unfreeze_memory_updates = lambda m: None

    results = []
    qa_count = 0

    for conv_idx, item in enumerate(data):
        conv_dict = item["conversation"]
        qa_list = item.get("qa", [])
        turns = extract_turns(conv_dict)

        for qa in qa_list:
            if args.max_qa and qa_count >= args.max_qa:
                return results

            question = qa["question"]
            reference = str(qa["answer"])
            cat_id = qa.get("category", 0)
            cat_name = CATEGORY_NAMES.get(cat_id, f"cat{cat_id}")

            if cat_id == 5 and not args.include_adversarial:
                continue

            # Last 10 turns as context
            recent = turns[-10:] if len(turns) > 10 else turns
            context = "\n".join(recent)
            prompt = (
                f"Conversation (recent):\n{context}\n\n"
                f"Answer briefly (under 30 words):\n"
                f"Q: {question}\nA:"
            )
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(args.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )
            new_tokens = out[0, inputs["input_ids"].shape[1]:]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            for stop in ("\n\n", "\nQuestion", "\nQ:"):
                idx = prediction.find(stop)
                if idx != -1:
                    prediction = prediction[:idx].strip()

            score = judge.score(question, reference, prediction) if judge else token_f1(prediction, reference)

            results.append({
                "conv_idx": conv_idx,
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "category": cat_name,
                "score": score,
            })
            qa_count += 1

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(results: list[dict]) -> dict:
    cat_scores: dict[str, list[float]] = defaultdict(list)
    all_scores = []
    for r in results:
        all_scores.append(r["score"])
        cat_scores[r["category"]].append(r["score"])

    return {
        "total": len(all_scores),
        "overall": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "by_category": {
            cat: {"n": len(s), "avg": sum(s) / len(s)}
            for cat, s in sorted(cat_scores.items())
        },
    }


def print_summary(summary: dict, mode: str, metric: str) -> None:
    print("\n" + "=" * 60)
    print(f"LoCoMo-10 评测结果  [{mode}]  metric={metric}")
    print(f"  样本数   : {summary['total']}")
    print(f"  Overall  : {summary['overall']*100:.1f}%")
    for cat, stat in summary["by_category"].items():
        print(f"  {cat:<18}: n={stat['n']:4d}  avg={stat['avg']*100:.1f}%")
    print(f"\n  mem0 参考 (LLM judge): 66.9%")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="locomo10.json path")
    p.add_argument("--oracle", default=None, help="MemoryOracle checkpoint dir (stage3/)")
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--base-model",
                   default="/home/shuzuan/models/Qwen/Qwen3___5-0.8B")
    p.add_argument("--device", default="cuda:7")
    p.add_argument("--max-qa", type=int, default=None,
                   help="Limit number of QA pairs (e.g. 20 for quick test)")
    p.add_argument("--commit-every", type=int, default=30,
                   help="Commit NLM state every N turns (keeps write buffer within training context; 0=disable)")
    p.add_argument("--judge", default=None,
                   help="LLM judge model name (e.g. gpt-4o-mini); None = F1 fallback")
    p.add_argument("--judge-base-url", default=None)
    p.add_argument("--include-adversarial", action="store_true",
                   help="Include category 5 (adversarial / I-don't-know) questions")
    p.add_argument("--output", default=None, help="Save JSON results to file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data = load_locomo10(args.data)
    log.info("Loaded %d conversations from %s", len(data), args.data)

    judge: LLMJudge | None = None
    if args.judge:
        judge = LLMJudge(args.judge, args.judge_base_url)
        metric = f"LLM-judge ({args.judge})"
    else:
        metric = "token-F1 (fallback)"
    log.info("Metric: %s", metric)

    if args.baseline_only:
        results = eval_baseline(data, args, judge)
        mode = "baseline"
    else:
        if not args.oracle:
            log.error("Provide --oracle or --baseline-only")
            sys.exit(1)
        results = eval_oracle(data, args, judge)
        mode = "oracle"

    summary = summarize(results)
    print_summary(summary, mode, metric)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump({"mode": mode, "metric": metric, "summary": summary,
                       "results": results}, f, ensure_ascii=False, indent=2)
        log.info("Saved → %s", args.output)


if __name__ == "__main__":
    main()
