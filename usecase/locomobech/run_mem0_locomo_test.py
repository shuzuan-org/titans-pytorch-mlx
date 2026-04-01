#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from urllib import error, request

from usecase.locomobech.common import CATEGORY_NAMES, load_locomo10, save_json
from usecase.locomobech.judge_client import MiniMaxJudge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mem0 LoCoMo memory test.")
    parser.add_argument("--data-file", default="data/dataset/locomo/locomo10.json")
    parser.add_argument("--base-url", default="http://58.211.6.130:10281")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", default="results/locomobech/mem0_predictions.json")
    parser.add_argument("--max-qa", type=int, default=None)
    parser.add_argument("--judge-base-url", default="https://mini.origintask.cn")
    parser.add_argument("--judge-model", default="glm-4.7")
    parser.add_argument("--judge-api-key", default="shuzuan2025-minimax")
    return parser.parse_args()


def post_json(base_url: str, path: str, payload: dict) -> dict:
    req = request.Request(
        base_url.rstrip("/") + "/" + path.lstrip("/"),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {payload}") from exc


def extract_memories(search_payload: dict) -> list[str]:
    data = search_payload.get("data", {})
    results = data.get("results", [])
    if not isinstance(results, list) or not results:
        return []
    pieces = []
    for item in results[:5]:
        if isinstance(item, dict):
            text = item.get("memory") or item.get("text") or item.get("content") or ""
            if text:
                pieces.append(str(text).strip())
    return pieces


def main() -> None:
    args = parse_args()
    data = load_locomo10(args.data_file)
    judge = MiniMaxJudge(args.judge_base_url, args.judge_api_key, args.judge_model)
    output = {}
    qa_count = 0
    for conv_idx, item in enumerate(data):
        user_id = f"locomo_conv_{conv_idx}"
        rows = []
        for qa in item.get("qa", []):
            category = int(qa.get("category", 0))
            if category == 5:
                continue
            question = str(qa.get("question", "")).strip()
            answer = str(qa.get("answer", "")).strip()
            search_result = post_json(
                args.base_url,
                f"/api/v1/users/{user_id}/search",
                {"query": question, "limit": args.limit},
            )
            memories = extract_memories(search_result)
            response_text = judge.answer_from_memories(question, memories) if memories else ""
            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "response": response_text,
                    "category": CATEGORY_NAMES.get(category, str(category)),
                    "retrieved_memories": memories,
                    "raw_search": search_result,
                }
            )
            qa_count += 1
            if args.max_qa and qa_count >= args.max_qa:
                break
        output[str(conv_idx)] = rows
        if args.max_qa and qa_count >= args.max_qa:
            break
    save_json(args.output, output)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()


