#!/usr/bin/env python3
# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Convert Wikipedia XML dumps to JSONL files.

Depends on wikiextractor (pip install wikiextractor).
Processes multiple language dumps in parallel.

Input:  ~/data/wiki_multilingual/dumps/{lang}wiki-*.xml.bz2
Output: /home/shuzuan/data/wiki_multilingual/{lang}/*.jsonl

Usage:
    python scripts/convert_wiki.py \
      --dumps-dir /home/shuzuan/data/wiki_multilingual/dumps \
      --output-dir /home/shuzuan/data/wiki_multilingual \
      --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MIN_ARTICLE_CHARS = 1000  # Skip stub articles


def check_wikiextractor() -> None:
    """Verify wikiextractor is installed and works."""
    try:
        import wikiextractor
        import wikiextractor.extract
    except Exception as e:
        logger.error(f"wikiextractor import failed: {e}")
        logger.error("Fix: pip install --upgrade wikiextractor")
        logger.error("Or manually patch: edit wikiextractor/extract.py line 378-379")
        logger.error("  Replace '(?i)' with '', change 're.S | re.U)' to 're.S | re.U | re.I)'")
        sys.exit(1)


def extract_lang_from_filename(filename: str) -> str | None:
    """Extract language code from dump filename like 'enwiki-20240101-pages-articles.xml.bz2'."""
    stem = Path(filename).name
    if "wiki-" in stem:
        return stem.split("wiki-")[0]
    return None


def convert_dump(args: tuple) -> tuple[str, int, int]:
    """Convert a single Wikipedia dump bz2 to JSONL.

    Returns (lang, article_count, token_estimate).
    """
    dump_path, output_dir, lang = args

    output_lang_dir = Path(output_dir) / lang
    output_lang_dir.mkdir(parents=True, exist_ok=True)

    done_marker = output_lang_dir / ".done"
    if done_marker.exists():
        # Count existing articles
        count = 0
        for f in output_lang_dir.glob("*.jsonl"):
            with open(f, encoding="utf-8") as fp:
                count += sum(1 for _ in fp)
        logger.info(f"[{lang}] Already done ({count} articles), skipping")
        return lang, count, 0

    logger.info(f"[{lang}] Extracting {dump_path} ...")

    # Use wikiextractor to extract text to a temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            sys.executable,
            "-m",
            "wikiextractor.WikiExtractor",
            "--output",
            tmp_dir,
            "--json",
            "--no-templates",
            "--processes",
            "4",
            "--quiet",
            str(dump_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"[{lang}] wikiextractor failed: {result.stderr[:500]}")
            return lang, 0, 0

        # Collect all extracted articles into a single JSONL
        article_count = 0
        output_file = output_lang_dir / "wiki.jsonl"

        with open(output_file, "w", encoding="utf-8") as out_f:
            for part_file in sorted(Path(tmp_dir).rglob("wiki_*")):
                if not part_file.is_file():
                    continue
                with open(part_file, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            text = obj.get("text", "")
                            if len(text) < MIN_ARTICLE_CHARS:
                                continue
                            # Write simplified record
                            out_f.write(
                                json.dumps(
                                    {
                                        "text": text,
                                        "title": obj.get("title", ""),
                                        "lang": lang,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            article_count += 1
                        except (json.JSONDecodeError, KeyError):
                            pass

    done_marker.touch()
    logger.info(f"[{lang}] Done: {article_count} articles → {output_file}")
    return lang, article_count, article_count * 500  # rough token estimate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Wikipedia XML dumps to JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dumps-dir",
        type=str,
        default="/home/shuzuan/data/wiki_multilingual/dumps",
        help="Directory containing *.xml.bz2 dump files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/shuzuan/data/wiki_multilingual",
        help="Output directory for per-language JSONL files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel dump processors",
    )
    parser.add_argument(
        "--langs",
        nargs="*",
        default=None,
        help="Languages to process (default: all found in dumps-dir)",
    )
    args = parser.parse_args()

    check_wikiextractor()

    dumps_dir = Path(args.dumps_dir)
    if not dumps_dir.exists():
        logger.error(f"Dumps directory not found: {dumps_dir}")
        sys.exit(1)

    # Find all dump files
    dump_files = sorted(dumps_dir.glob("*.xml.bz2"))
    if not dump_files:
        logger.error(f"No .xml.bz2 files found in {dumps_dir}")
        sys.exit(1)

    # Map lang → dump file
    lang_dumps = {}
    for df in dump_files:
        lang = extract_lang_from_filename(df.name)
        if lang:
            lang_dumps[lang] = df

    if args.langs:
        lang_dumps = {l: p for l, p in lang_dumps.items() if l in args.langs}

    logger.info(f"Found {len(lang_dumps)} language dumps: {sorted(lang_dumps.keys())}")

    # Process dumps in parallel
    worker_args = [
        (str(path), args.output_dir, lang) for lang, path in lang_dumps.items()
    ]

    total_articles = 0
    with ProcessPoolExecutor(max_workers=min(args.workers, len(worker_args))) as exe:
        futures = {exe.submit(convert_dump, a): a[2] for a in worker_args}
        for future in as_completed(futures):
            lang = futures[future]
            try:
                _, count, _ = future.result()
                total_articles += count
            except Exception as e:
                logger.error(f"[{lang}] Failed: {e}")

    logger.info("=" * 60)
    logger.info(f"Wikipedia conversion complete!")
    logger.info(f"  Total articles: {total_articles:,}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
