#!/usr/bin/env python3
# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Pre-tokenize local JSONL files to binary uint32 .bin shards.

Reads .jsonl / .jsonl.gz files from multiple source directories,
tokenizes the `text` field using Qwen2 tokenizer, and writes
numpy uint32 binary shards (512M tokens each).

Features:
- Multi-process tokenization (default: 32 workers)
- Resumable via .progress checkpoint files (per-file, not per-merge)
- Minimum text length filtering (configurable via --min-chars)
- Cross-shard token packing (no wasted space)

Usage:
    python scripts/pretokenize_local.py \
      --sources /home/shuzuan/data/longwanjuan/en \
                /home/shuzuan/data/books_en \
      --output /home/shuzuan/tokens/en \
      --tokenizer Qwen/Qwen2-7B \
      --workers 32
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
import tarfile
import time
from pathlib import Path
from typing import Iterator

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Tokens per shard (512M × 4 bytes = 2GB per shard)
TOKENS_PER_SHARD = 512 * 1024 * 1024
MIN_CHARS = 8192  # Default, overrideable via --min-chars

# Per-worker global tokenizer (initialized once per process via Pool initializer)
_worker_tokenizer = None


def _init_worker(tokenizer_name: str) -> None:
    """Pool initializer: load tokenizer once per worker process."""
    global _worker_tokenizer
    from transformers import AutoTokenizer

    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def iter_data_files(source_dirs: list[Path]) -> Iterator[Path]:
    """Yield all supported data files from source directories.

    Supported formats:
    - .jsonl          plain JSONL
    - .jsonl.gz       gzipped JSONL
    - .jsonl.tar.gz   tar-gzipped JSONL (WanJuan style)
    - .parquet        Apache Parquet (Gutenberg/Wikipedia HF style)
    """
    patterns = ("**/*.jsonl", "**/*.jsonl.gz", "**/*.jsonl.tar.gz", "**/*.parquet")
    for src in source_dirs:
        src = Path(src)
        if not src.exists():
            logger.warning(f"Source directory not found: {src}")
            continue
        seen: set[Path] = set()
        for pattern in patterns:
            for f in sorted(src.glob(pattern)):
                if f not in seen:
                    seen.add(f)
                    yield f


def _tok_filename(filepath: str) -> str:
    """Derive a collision-free .tok filename from the full file path."""
    h = hashlib.md5(filepath.encode()).hexdigest()[:12]
    stem = Path(filepath).stem[:32]  # truncate for readability
    return f"{stem}_{h}.tok"


def _iter_jsonl_lines(fp, min_chars: int) -> Iterator[str]:
    """Yield texts from an open JSONL text stream."""
    for line in fp:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            text = obj.get("text", obj.get("content", obj.get("passage", "")))
            if isinstance(text, str) and len(text) >= min_chars:
                yield text
        except json.JSONDecodeError:
            pass


def _iter_texts(filepath: Path, min_chars: int) -> Iterator[str]:
    """Yield texts from a data file one at a time (streaming).

    Dispatches on file extension:
    - .parquet        → pyarrow row-group streaming (columns: text or content)
    - .jsonl.tar.gz   → tar extraction then JSONL streaming
    - .jsonl.gz       → gzip then JSONL streaming
    - .jsonl          → plain JSONL streaming
    """
    name = filepath.name

    if name.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(filepath))
        text_col = None
        for col in ("text", "content", "passage"):
            if col in pf.schema_arrow.names:
                text_col = col
                break
        if text_col is None:
            return
        for batch in pf.iter_batches(columns=[text_col], batch_size=256):
            for val in batch.column(0):
                text = val.as_py()
                if isinstance(text, str) and len(text) >= min_chars:
                    yield text

    elif name.endswith(".jsonl.tar.gz"):
        with tarfile.open(str(filepath), "r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                fp = tar.extractfile(member)
                if fp is None:
                    continue
                with fp:
                    text_stream = (line.decode("utf-8", errors="replace") for line in fp)
                    yield from _iter_jsonl_lines(text_stream, min_chars)

    elif name.endswith(".jsonl.gz"):
        with gzip.open(str(filepath), "rt", encoding="utf-8", errors="replace") as f:
            yield from _iter_jsonl_lines(f, min_chars)

    else:  # plain .jsonl
        with open(str(filepath), "rt", encoding="utf-8", errors="replace") as f:
            yield from _iter_jsonl_lines(f, min_chars)


def tokenize_file_worker(args: tuple) -> tuple[str, int]:
    """Tokenize one file and return (filepath, token_count).

    Streams text line-by-line and writes uint32 tokens directly to the .tok
    file — no intermediate Python list of ints, no full-file RAM load.
    """
    filepath, output_dir, min_chars = args
    tok_file = Path(output_dir) / "tmp" / _tok_filename(filepath)

    if tok_file.exists():
        # Already processed; derive count from file size (uint32 = 4 bytes each)
        return filepath, tok_file.stat().st_size // 4

    tok_file.parent.mkdir(parents=True, exist_ok=True)
    eos_id = _worker_tokenizer.eos_token_id
    total = 0

    try:
        with open(tok_file, "wb") as out:
            for text in _iter_texts(Path(filepath), min_chars):
                ids = _worker_tokenizer.encode(text, add_special_tokens=False)
                arr = np.array(ids, dtype=np.uint32)
                arr.tofile(out)
                total += len(arr)
                if eos_id is not None:
                    np.array([eos_id], dtype=np.uint32).tofile(out)
                    total += 1
    except Exception as e:
        logger.warning(f"Error processing {filepath}: {e}")
        tok_file.unlink(missing_ok=True)
        return filepath, 0

    # Empty file = empty marker (tok_file exists, size 0)
    return filepath, total


def merge_tok_files_to_shards(
    tok_dir: Path,
    output_dir: Path,
    tokens_per_shard: int = TOKENS_PER_SHARD,
) -> int:
    """Merge all .tok files into fixed-size .bin shards.

    Always regenerates from scratch — tokenization is checkpointed at the
    .tok file level, so merge is cheap and idempotent.
    """
    tok_files = sorted(tok_dir.glob("*.tok"))
    if not tok_files:
        logger.warning("No .tok files to merge")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove any existing shards to avoid stale data
    for old in output_dir.glob("shard_*.bin"):
        old.unlink()

    shard_idx = 0
    buffer = np.empty(tokens_per_shard, dtype=np.uint32)
    buf_pos = 0
    total_tokens = 0

    def flush_shard() -> None:
        nonlocal shard_idx, buf_pos
        if buf_pos == 0:
            return
        shard_path = output_dir / f"shard_{shard_idx:05d}.bin"
        buffer[:buf_pos].tofile(str(shard_path))
        logger.info(
            f"  Wrote shard {shard_idx:05d}: {buf_pos / 1e6:.1f}M tokens → {shard_path}"
        )
        shard_idx += 1
        buf_pos = 0

    for tok_file in tok_files:
        file_size = tok_file.stat().st_size
        if file_size == 0:
            continue  # empty marker for files that produced no tokens

        chunk = np.fromfile(str(tok_file), dtype=np.uint32)
        total_tokens += len(chunk)
        offset = 0

        while offset < len(chunk):
            space = tokens_per_shard - buf_pos
            take = min(space, len(chunk) - offset)
            buffer[buf_pos : buf_pos + take] = chunk[offset : offset + take]
            buf_pos += take
            offset += take

            if buf_pos == tokens_per_shard:
                flush_shard()

    # Final partial shard
    if buf_pos > 0:
        flush_shard()

    return total_tokens


def load_progress(progress_file: Path) -> set[str]:
    """Load set of already-processed file paths from JSON."""
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_progress(progress_file: Path, done: set[str]) -> None:
    """Save progress checkpoint atomically as JSON (write then rename)."""
    tmp = progress_file.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sorted(done), f)
    tmp.replace(progress_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize local JSONL files to binary uint32 shards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Source directories containing .jsonl / .jsonl.gz files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for .bin shards",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2-7B",
        help="HuggingFace tokenizer to use",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--tokens-per-shard",
        type=int,
        default=TOKENS_PER_SHARD,
        help="Tokens per output shard (default: 512M)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=MIN_CHARS,
        help="Minimum text length to include",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    progress_file = output_dir / ".progress"
    done_files = load_progress(progress_file)

    # Collect all JSONL files
    source_dirs = [Path(s) for s in args.sources]
    all_files = list(iter_data_files(source_dirs))
    pending = [str(f) for f in all_files if str(f) not in done_files]

    logger.info(f"Found {len(all_files)} JSONL files total, {len(pending)} pending")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Tokenizer: {args.tokenizer}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Min chars: {args.min_chars}")

    if pending:
        # min_chars is passed into each worker via the args tuple
        worker_args = [(fp, str(output_dir), args.min_chars) for fp in pending]

        start_time = time.time()
        total_new_tokens = 0

        # spawn avoids CUDA fork issues; tokenizer initialized once per worker
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.tokenizer,),
        ) as pool:
            for i, (filepath, token_count) in enumerate(
                pool.imap_unordered(tokenize_file_worker, worker_args, chunksize=4)
            ):
                done_files.add(filepath)
                total_new_tokens += token_count

                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (len(pending) - i - 1) / rate if rate > 0 else 0
                    logger.info(
                        f"  [{i+1}/{len(pending)}] "
                        f"{total_new_tokens/1e9:.2f}B tokens processed, "
                        f"ETA: {remaining/60:.1f} min"
                    )

                if (i + 1) % 50 == 0:
                    save_progress(progress_file, done_files)

        save_progress(progress_file, done_files)
        elapsed = time.time() - start_time
        logger.info(
            f"Tokenization complete: {total_new_tokens/1e9:.2f}B new tokens "
            f"in {elapsed/60:.1f} min"
        )
    else:
        logger.info("All files already tokenized. Proceeding to merge.")

    # Merge .tok files into .bin shards (always full re-merge for correctness)
    logger.info("Merging token files into shards (full re-merge)...")
    total_tokens = merge_tok_files_to_shards(tmp_dir, output_dir, args.tokens_per_shard)

    shards = list(output_dir.glob("shard_*.bin"))
    logger.info("=" * 60)
    logger.info("Pre-tokenization complete!")
    logger.info(f"  Total tokens: {total_tokens/1e9:.2f}B")
    logger.info(f"  Shards written: {len(shards)}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
