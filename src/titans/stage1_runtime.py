from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from titans.stage1_models import (
    FrozenBackboneWithTimelineMemory,
    Stage1ModelConfig,
    Stage1SessionState,
    build_stage1_model,
)


@dataclass
class Stage1WriteResult:
    session_id: str
    memory_version: int
    num_items_written: int
    profile: dict[str, float | int] = field(default_factory=dict)


@dataclass
class Stage1ChatResult:
    session_id: str
    answer: str
    memory_version: int
    retrieval_weights: torch.Tensor | None = None
    profile: dict[str, float | int | str] = field(default_factory=dict)


class Stage1SessionStore:
    def __init__(self) -> None:
        self._states: dict[str, Stage1SessionState] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _get_lock(self, session_id: str) -> threading.Lock:
        with self._global_lock:
            if session_id not in self._locks:
                self._locks[session_id] = threading.Lock()
            return self._locks[session_id]

    def get(self, session_id: str) -> Stage1SessionState | None:
        return self._states.get(session_id)

    def set(self, session_id: str, state: Stage1SessionState) -> None:
        self._states[session_id] = state

    def delete(self, session_id: str) -> bool:
        existed = session_id in self._states
        self._states.pop(session_id, None)
        return existed

    def metadata(self, session_id: str) -> dict[str, Any] | None:
        state = self._states.get(session_id)
        if state is None:
            return None
        return {
            "format_version": state.format_version,
            "session_id": state.session_id,
            "memory_version": state.memory_version,
            "model_signature": dict(state.model_signature),
            "stats": {
                "num_writes": state.stats.num_writes,
                "num_queries": state.stats.num_queries,
                "updated_at": state.stats.updated_at,
            },
        }

    def save_snapshot(
        self,
        session_id: str,
        path: str | Path,
        model: FrozenBackboneWithTimelineMemory,
    ) -> None:
        state = self._states[session_id]
        torch.save(model.serialize_session_state(state), Path(path))

    def load_snapshot(
        self,
        path: str | Path,
        model: FrozenBackboneWithTimelineMemory,
    ) -> Stage1SessionState:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        state = model.deserialize_session_state(payload)
        self.set(state.session_id, state)
        return state


class Stage1MemoryWriter:
    def __init__(
        self,
        model: FrozenBackboneWithTimelineMemory,
        store: Stage1SessionStore,
    ) -> None:
        self.model = model
        self.store = store

    def write(
        self,
        session_id: str,
        texts: list[str],
        idempotency_key: str | None = None,
    ) -> Stage1WriteResult:
        del idempotency_key
        if not texts:
            state = self.store.get(session_id) or self.model.init_session_state(session_id)
            self.store.set(session_id, state)
            return Stage1WriteResult(
                session_id=session_id,
                memory_version=state.memory_version,
                num_items_written=0,
            )

        lock = self.store._get_lock(session_id)
        with lock:
            state = self.store.get(session_id) or self.model.init_session_state(session_id)
            next_state, profile = self.model.write_texts(
                state,
                texts,
                updated_at=time.time(),
            )
            self.store.set(session_id, next_state)
            return Stage1WriteResult(
                session_id=session_id,
                memory_version=next_state.memory_version,
                num_items_written=len(texts),
                profile=profile,
            )


class Stage1ChatGenerator:
    def __init__(
        self,
        model: FrozenBackboneWithTimelineMemory,
        store: Stage1SessionStore,
    ) -> None:
        self.model = model
        self.store = store

    def chat(
        self,
        session_id: str,
        query: str,
        generation_config: dict[str, Any] | None = None,
        include_debug: bool = False,
    ) -> Stage1ChatResult:
        state = self.store.get(session_id)
        if state is None:
            state = self.model.init_session_state(session_id)
            self.store.set(session_id, state)

        result = self.model.answer_query(
            state,
            query,
            generation_config=generation_config,
            updated_at=time.time(),
        )
        updated_state = result["session_state"]
        self.store.set(session_id, updated_state)
        return Stage1ChatResult(
            session_id=session_id,
            answer=str(result["answer"]),
            memory_version=int(result["memory_version"]),
            retrieval_weights=result["retrieval_weights"] if include_debug else None,
            profile=dict(result.get("profile", {})),
        )

    def chat_direct(
        self,
        session_id: str,
        query: str,
        generation_config: dict[str, Any] | None = None,
    ) -> Stage1ChatResult:
        state = self.store.get(session_id)
        if state is None:
            state = self.model.init_session_state(session_id)
            self.store.set(session_id, state)

        result = self.model.answer_query_direct(
            query=query,
            generation_config=generation_config,
        )
        return Stage1ChatResult(
            session_id=session_id,
            answer=str(result["answer"]),
            memory_version=int(state.memory_version),
            retrieval_weights=None,
            profile=dict(result.get("profile", {})),
        )


class Stage1DeploymentRuntime:
    def __init__(
        self,
        model: FrozenBackboneWithTimelineMemory,
        store: Stage1SessionStore | None = None,
    ) -> None:
        self.model = model
        self.store = store or Stage1SessionStore()
        self.writer = Stage1MemoryWriter(model, self.store)
        self.chat_generator = Stage1ChatGenerator(model, self.store)

    @classmethod
    def from_model_config(
        cls,
        config: Stage1ModelConfig,
        checkpoint_path: str | Path | None = None,
    ) -> Stage1DeploymentRuntime:
        model = build_stage1_model(config)
        if checkpoint_path is not None:
            payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
            trainable_state_dict = payload.get("trainable_state_dict", payload)
            model.load_trainable_state_dict(trainable_state_dict)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return cls(model=model)

    def write_memory(
        self,
        session_id: str,
        content: str | None = None,
        contents: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> Stage1WriteResult:
        payload = contents if contents is not None else ([content] if content is not None else [])
        return self.writer.write(session_id=session_id, texts=payload, idempotency_key=idempotency_key)

    def chat_with_memory(
        self,
        session_id: str,
        query: str,
        generation_config: dict[str, Any] | None = None,
        include_debug: bool = False,
    ) -> Stage1ChatResult:
        return self.chat_generator.chat(
            session_id=session_id,
            query=query,
            generation_config=generation_config,
            include_debug=include_debug,
        )

    def chat_direct_backbone(
        self,
        session_id: str,
        query: str,
        generation_config: dict[str, Any] | None = None,
    ) -> Stage1ChatResult:
        return self.chat_generator.chat_direct(
            session_id=session_id,
            query=query,
            generation_config=generation_config,
        )

    def delete_session(self, session_id: str) -> bool:
        return self.store.delete(session_id)

    def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        return self.store.metadata(session_id)

    def save_session_snapshot(self, session_id: str, path: str | Path) -> None:
        self.store.save_snapshot(session_id, path, self.model)

    def load_session_snapshot(self, path: str | Path) -> Stage1SessionState:
        return self.store.load_snapshot(path, self.model)

