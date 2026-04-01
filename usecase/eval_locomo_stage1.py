#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib import error, request


CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-ended",
    5: "adversarial",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stage1 service on LoCoMo-10.")
    parser.add_argument("--data-file", default="data/dataset/locomo/locomo10.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-qa", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--include-adversarial", action="store_true")
    return parser.parse_args()


def read_json(req: request.Request, timeout: int) -> dict[str, Any]:
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {payload}") from exc


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        base_url.rstrip("/") + "/" + path.lstrip("/"),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return read_json(req, timeout=timeout)


def delete_json(base_url: str, path: str, timeout: int) -> dict[str, Any]:
    req = request.Request(
        base_url.rstrip("/") + "/" + path.lstrip("/"),
        method="DELETE",
    )
    return read_json(req, timeout=timeout)


def load_locomo10(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("locomo10 data must be a list")
    return payload


def extract_turns(conv_dict: dict[str, Any]) -> list[str]:
    turns: list[str] = []
    for index in range(1, 100):
        key = f"session_{index}"
        if key not in conv_dict:
            break
        session = conv_dict[key]
        if not isinstance(session, list):
            continue
        for turn in session:
            if isinstance(turn, dict):
                speaker = str(turn.get("speaker", "")).strip()
                text = str(turn.get("text", "")).strip()
                if text:
                    turns.append(f"{speaker}: {text}" if speaker else text)
            elif isinstance(turn, str) and turn.strip():
                turns.append(turn.strip())
    return turns


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(str(gold).lower().split())
    if not pred_tokens or not gold_tokens:
        return float(pred.strip().lower() == str(gold).strip().lower())
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def truncate_answer(text: str) -> str:
    text = text.strip()
    for stop in ("\n\n", "\nQuestion", "\nQ:", "Answer:"):
        index = text.find(stop)
        if index != -1:
            text = text[:index].strip()
    return text


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    cat_scores: dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []
    for item in results:
        score = float(item["score"])
        all_scores.append(score)
        cat_scores[item["category"]].append(score)
    return {
        "total": len(all_scores),
        "overall": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "by_category": {
            cat: {"n": len(scores), "avg": sum(scores) / len(scores)}
            for cat, scores in sorted(cat_scores.items())
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("LoCoMo-10 stage1 eval  [service]  metric=token-F1")
    print(f"  样本数   : {summary['total']}")
    print(f"  Overall  : {summary['overall']*100:.1f}%")
    for cat, stat in summary["by_category"].items():
        print(f"  {cat:<18}: n={stat['n']:4d}  avg={stat['avg']*100:.1f}%")
    print("\n  mem0 参考 (LLM judge): 66.9%")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    data = load_locomo10(args.data_file)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.temperature > 0,
    }

    results: list[dict[str, Any]] = []
    qa_count = 0
    for conv_idx, item in enumerate(data):
        session_id = f"locomo-stage1-{conv_idx}"
        turns = extract_turns(item["conversation"])
        write_response = post_json(
            args.base_url,
            "/v1/memory/write",
            {"session_id": session_id, "contents": turns},
            timeout=args.request_timeout,
        )
        for qa in item.get("qa", []):
            category_id = int(qa.get("category", 0))
            if category_id == 5 and not args.include_adversarial:
                continue
            question = str(qa.get("question", "")).strip()
            reference = str(qa.get("answer", "")).strip()
            if not question or not reference:
                continue
            response = post_json(
                args.base_url,
                "/v1/chat/respond",
                {
                    "session_id": session_id,
                    "query": question,
                    "generation_config": generation_config,
                },
                timeout=args.request_timeout,
            )
            prediction = truncate_answer(str(response.get("answer", "")))
            score = token_f1(prediction, reference)
            results.append(
                {
                    "conv_idx": conv_idx,
                    "question": question,
                    "reference": reference,
                    "prediction": prediction,
                    "category": CATEGORY_NAMES.get(category_id, f"cat{category_id}"),
                    "score": score,
                    "write_profile": write_response.get("profile", {}),
                    "chat_profile": response.get("profile", {}),
                }
            )
            qa_count += 1
            if qa_count % 20 == 1 or qa_count == 1:
                avg = sum(item["score"] for item in results) / len(results)
                print(f"[{qa_count:3d}] {CATEGORY_NAMES.get(category_id, category_id)} avg={avg:.3f}  Q: {question[:60]}")
                print(f"  pred: {prediction[:100]}")
                print(f"  gold: {reference[:100]}")
            if args.max_qa and qa_count >= args.max_qa:
                break
        delete_json(args.base_url, f"/v1/sessions/{session_id}", timeout=args.request_timeout)
        if args.max_qa and qa_count >= args.max_qa:
            break

    summary = summarize(results)
    print_summary(summary)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "mode": "service",
                    "metric": "token-F1",
                    "summary": summary,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved to {output_path}")


if __name__ == "__main__":
    main()

