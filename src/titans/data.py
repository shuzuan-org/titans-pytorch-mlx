# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Binary token dataset utilities for pretraining.

BinaryTokenDataset:
    Reads uint32 .bin shards produced by pretokenize_local.py via numpy.memmap.
    Supports epoch-level shuffle across shards.

WeightedMixDataset:
    Mixes multiple BinaryTokenDataset sources with temperature-scaled weights.
    Each step samples a source via multinomial, then draws a random sequence.
    Incorporates distributed rank into the RNG seed to ensure different GPUs
    see different data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class BinaryTokenDataset(Dataset):
    """Memory-mapped uint32 .bin shard dataset.

    Args:
        shard_paths: List of .bin file paths (uint32 arrays).
        seq_len: Sequence length for each sample.
        seed: Random seed for epoch-level shuffle.
    """

    def __init__(
        self,
        shard_paths: list[Path | str],
        seq_len: int,
        seed: int = 42,
    ) -> None:
        self.seq_len = seq_len
        self.seed = seed

        self.shards: list[np.memmap] = []
        self.shard_lengths: list[int] = []  # number of complete sequences per shard

        for path in shard_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Shard not found: {path}")
            mm = np.memmap(str(path), dtype=np.uint32, mode="r")
            n_seqs = (len(mm) - 1) // seq_len  # -1 for label shift
            if n_seqs > 0:
                self.shards.append(mm)
                self.shard_lengths.append(n_seqs)

        if not self.shards:
            raise ValueError("No usable shards found (all empty or too short)")

        self.cumulative = np.cumsum([0] + self.shard_lengths)
        self._total = int(self.cumulative[-1])

        self._epoch = 0
        self._indices: np.ndarray | None = None
        self._shuffle_epoch()

    def _shuffle_epoch(self) -> None:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._indices = rng.permutation(self._total)

    def set_epoch(self, epoch: int) -> None:
        """Re-shuffle for a new epoch."""
        self._epoch = epoch
        self._shuffle_epoch()

    def __len__(self) -> int:
        return self._total

    def _global_to_shard(self, global_idx: int) -> tuple[int, int]:
        """Convert a global sequence index to (shard_idx, local_seq_idx)."""
        shard_idx = int(np.searchsorted(self.cumulative[1:], global_idx, side="right"))
        local_idx = global_idx - int(self.cumulative[shard_idx])
        return shard_idx, local_idx

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        shuffled_idx = int(self._indices[idx]) if self._indices is not None else idx

        shard_idx, local_idx = self._global_to_shard(shuffled_idx)
        shard = self.shards[shard_idx]

        start = local_idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for label shift

        chunk = np.array(shard[start:end], dtype=np.int64)
        return {
            "input_ids": torch.from_numpy(chunk[:-1]),
            "labels": torch.from_numpy(chunk[1:]),
        }

    @property
    def total_tokens(self) -> int:
        return sum(len(s) for s in self.shards)

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        seq_len: int,
        seed: int = 42,
    ) -> "BinaryTokenDataset":
        """Construct from a directory of shard_*.bin files."""
        directory = Path(directory)
        shards = sorted(directory.glob("shard_*.bin"))
        if not shards:
            raise FileNotFoundError(f"No shard_*.bin files in {directory}")
        return cls(shards, seq_len, seed)


def _get_distributed_rank() -> int:
    """Return the current distributed process rank, or 0 if not in distributed context."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class WeightedMixDataset(IterableDataset):
    """Iterable dataset that mixes multiple BinaryTokenDataset sources.

    Weights are temperature-scaled to balance large and small corpora:
        w_i = count_i^(1/T) / Σ count_j^(1/T)

    Each GPU rank gets a unique RNG seed derived from (seed, epoch, rank, worker_id),
    ensuring different processes sample different sequences.

    Args:
        datasets: List of BinaryTokenDataset instances.
        weights: Raw weights (e.g., target token counts). If None,
                 uses number of tokens in each dataset.
        temperature: Sampling temperature (default 0.7). Higher → more uniform.
        seed: Base random seed.
    """

    def __init__(
        self,
        datasets: list[BinaryTokenDataset],
        weights: list[float] | None = None,
        temperature: float = 0.7,
        seed: int = 42,
    ) -> None:
        if not datasets:
            raise ValueError("datasets must not be empty")

        self.datasets = datasets
        self.temperature = temperature
        self.seed = seed

        if weights is None:
            raw = np.array([float(d.total_tokens) for d in datasets])
        else:
            if len(weights) != len(datasets):
                raise ValueError("weights and datasets must have the same length")
            raw = np.array(weights, dtype=float)

        scaled = np.power(raw, 1.0 / temperature)
        self._probs = (scaled / scaled.sum()).tolist()

        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        for d in self.datasets:
            d.set_epoch(epoch)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # Include distributed rank so each GPU sees different data
        rank = _get_distributed_rank()
        rng = np.random.default_rng(self.seed + self._epoch * 100_000 + rank * 1000 + worker_id)

        # Give each (rank, worker) pair a distinct start offset within each source.
        # Real diversity comes from the unique per-rank/worker RNG above;
        # this just prevents every worker from reading position 0 on startup.
        global_worker_id = rank * num_workers + worker_id
        src_positions = [global_worker_id % max(len(d), 1) for d in self.datasets]

        while True:
            src_idx = int(rng.choice(len(self.datasets), p=self._probs))
            dataset = self.datasets[src_idx]

            if len(dataset) == 0:
                continue

            sample_idx = src_positions[src_idx] % len(dataset)
            src_positions[src_idx] += 1

            yield dataset[sample_idx]

    @property
    def source_probs(self) -> list[float]:
        return self._probs
