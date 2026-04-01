#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from usecase.locomobech.common import save_json, summarize_scores, token_f1
from usecase.locomobech.judge_client import MiniMaxJudge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LoCoMo predictions with BLEU/F1/Judge.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-output", required=True)
    parser.add_argument("--judge-base-url", default="https://mini.origintask.cn")
    parser.add_argument("--judge-model", default="glm-4.7")
    parser.add_argument("--judge-api-key", default="shuzuan2025-minimax")
    return parser.parse_args()


def bleu1(prediction: str, reference: str) -> float:
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    ref_counts = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))
    return overlap / len(pred_tokens)


def main() -> None:
    args = parse_args()
    judge = MiniMaxJudge(
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
        model=args.judge_model,
    )
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    evaluated = {}
    rows = []
    for key, items in payload.items():
        evaluated_items = []
        for item in items:
            prediction = str(item.get("response", "")).strip()
            reference = str(item.get("answer", "")).strip()
            question = str(item.get("question", "")).strip()
            category = str(item.get("category", "unknown"))
            bleu_score = bleu1(prediction, reference)
            f1_score = token_f1(prediction, reference)
            llm_score = judge.score(question, reference, prediction)
            enriched = dict(item)
            enriched["bleu_score"] = bleu_score
            enriched["f1_score"] = f1_score
            enriched["llm_score"] = llm_score
            evaluated_items.append(enriched)
            rows.append(enriched)
        evaluated[key] = evaluated_items
    save_json(args.output, evaluated)
    save_json(args.score_output, summarize_scores(rows))
    print(f"saved eval to {args.output}")
    print(f"saved scores to {args.score_output}")


if __name__ == "__main__":
    main()


