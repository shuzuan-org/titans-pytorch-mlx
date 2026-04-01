#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from usecase.locomobech.common import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare mem0 and titans LoCoMo scores.")
    parser.add_argument("--mem0-scores", required=True)
    parser.add_argument("--titans-scores", required=True)
    parser.add_argument("--json-output", default="results/locomobech/compare_report.json")
    parser.add_argument("--md-output", default="results/locomobech/compare_report.md")
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    mem0_scores = load_json(args.mem0_scores)
    titans_scores = load_json(args.titans_scores)
    report = {
        "mem0": mem0_scores,
        "titans": titans_scores,
    }
    save_json(args.json_output, report)

    lines = [
        "# LoCoMo Compare Report",
        "",
        "## Overall",
        "",
        f"- mem0 BLEU: {mem0_scores['overall']['bleu_score']:.4f}",
        f"- mem0 F1: {mem0_scores['overall']['f1_score']:.4f}",
        f"- mem0 LLM: {mem0_scores['overall']['llm_score']:.4f}",
        f"- titans BLEU: {titans_scores['overall']['bleu_score']:.4f}",
        f"- titans F1: {titans_scores['overall']['f1_score']:.4f}",
        f"- titans LLM: {titans_scores['overall']['llm_score']:.4f}",
        "",
        "## By Category",
        "",
    ]
    categories = sorted(set(mem0_scores.get("by_category", {})) | set(titans_scores.get("by_category", {})))
    for category in categories:
        mem0_item = mem0_scores.get("by_category", {}).get(category, {})
        titans_item = titans_scores.get("by_category", {}).get(category, {})
        lines.extend(
            [
                f"### {category}",
                f"- mem0: {mem0_item}",
                f"- titans: {titans_item}",
                "",
            ]
        )
    Path(args.md_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md_output).write_text("\n".join(lines), encoding="utf-8")
    print(f"saved json to {args.json_output}")
    print(f"saved md to {args.md_output}")


if __name__ == "__main__":
    main()



