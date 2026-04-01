from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None

from titans.stage1_prompting import (
    DEFAULT_STAGE1_PROMPT_VERSION,
    build_stage1_question_prompt,
)


@dataclass
class Stage1ModelConfig:
    backbone_name: str = "Qwen/Qwen2.5-7B-Instruct"
    torch_dtype: str = "auto"
    attn_implementation: str | None = None
    history_backbone_mode: str = "full"
    memory_update_source: str = "last_hidden"
    num_retrieved_memory_tokens: int = 16
    loss_mask_scope: str = "answer_only"
    memory_slots: int = 16
    memory_hidden_mult: float = 2.0
    memory_dropout: float = 0.0
    trust_remote_code: bool = False
    prompt_version: str = DEFAULT_STAGE1_PROMPT_VERSION
    use_write_gate_loss: bool = False


@dataclass
class Stage1SessionStats:
    num_writes: int = 0
    num_queries: int = 0
    updated_at: float = 0.0


@dataclass
class Stage1SessionState:
    format_version: str
    session_id: str
    memory_version: int
    model_signature: dict[str, Any]
    memory_state: torch.Tensor
    stats: Stage1SessionStats


class FrozenBackboneAdapter(nn.Module):
    def __init__(self, config: Stage1ModelConfig) -> None:
        super().__init__()
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError(
                "transformers is required for stage1 training. Install titans[train]."
            )

        model_kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
        if config.torch_dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, config.torch_dtype)
        if config.attn_implementation:
            model_kwargs["attn_implementation"] = config.attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(
            config.backbone_name,
            **model_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.backbone_name,
            trust_remote_code=config.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.freeze_parameters()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def freeze_parameters(self) -> None:
        self.model.requires_grad_(False)
        self.model.eval()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def encode_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        return outputs.hidden_states[-1]

    def forward_with_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Any:
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

    def generate_from_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        config = dict(generation_config or {})
        if "pad_token_id" not in config:
            config["pad_token_id"] = self.tokenizer.pad_token_id
        if "eos_token_id" not in config:
            config["eos_token_id"] = self.tokenizer.eos_token_id
        with torch.inference_mode():
            return self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **config,
            )

    def generate_from_input_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        config = dict(generation_config or {})
        if "pad_token_id" not in config:
            config["pad_token_id"] = self.tokenizer.pad_token_id
        if "eos_token_id" not in config:
            config["eos_token_id"] = self.tokenizer.eos_token_id
        with torch.inference_mode():
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **config,
            )


class DenseTimelineMemory(nn.Module):
    def __init__(self, hidden_size: int, memory_slots: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_slots = memory_slots
        self.initial_memory = nn.Parameter(torch.randn(memory_slots, hidden_size) * 0.02)
        self.write_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.write_value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.write_gate = nn.Linear(hidden_size, 1)
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.initial_memory.unsqueeze(0).expand(batch_size, -1, -1).to(device=device)

    def update(self, state: torch.Tensor, write_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_dtype = self.write_key.weight.dtype
        state = state.to(dtype=target_dtype)
        write_repr = write_repr.to(dtype=target_dtype)
        key = self.write_key(write_repr)
        value = self.write_value(write_repr)
        scores = torch.matmul(state, key.unsqueeze(-1)).squeeze(-1) / (self.hidden_size**0.5)
        weights = torch.softmax(scores, dim=-1)
        gate = torch.sigmoid(self.write_gate(write_repr))
        update = weights.unsqueeze(-1) * value.unsqueeze(1)
        next_state = state + gate.unsqueeze(-1) * update
        return next_state, gate.squeeze(-1)

    def retrieve(self, state: torch.Tensor, query_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_dtype = self.query_proj.weight.dtype
        state = state.to(dtype=target_dtype)
        query_repr = query_repr.to(dtype=target_dtype)
        query = self.query_proj(query_repr)
        scores = torch.matmul(state, query.unsqueeze(-1)).squeeze(-1) / (self.hidden_size**0.5)
        weights = torch.softmax(scores, dim=-1)
        retrieved = torch.sum(weights.unsqueeze(-1) * state, dim=1)
        return self.output_proj(retrieved), weights



class FrozenBackboneWithTimelineMemory(nn.Module):
    session_format_version = "stage1-session-v1"

    def __init__(self, config: Stage1ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = FrozenBackboneAdapter(config)
        self.memory = DenseTimelineMemory(
            hidden_size=self.backbone.hidden_size,
            memory_slots=config.memory_slots,
        )
        self.write_decision_head = nn.Linear(self.backbone.hidden_size, 1)
        if not self.config.use_write_gate_loss:
            self.write_decision_head.requires_grad_(False)

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def count_parameters(self) -> tuple[int, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return total, trainable

    def model_signature(self) -> dict[str, Any]:
        config_dict = asdict(self.config)
        config_hash = hashlib.sha256(
            repr(sorted(config_dict.items())).encode("utf-8")
        ).hexdigest()[:16]
        return {
            "backbone_name": self.config.backbone_name,
            "hidden_size": self.backbone.hidden_size,
            "memory_slots": self.config.memory_slots,
            "config_hash": config_hash,
        }

    def _pool_hidden(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def _tokenize_texts(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.backbone.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return (
            batch["input_ids"].to(self.backbone.device),
            batch["attention_mask"].to(self.backbone.device),
        )

    def init_session_state(self, session_id: str) -> Stage1SessionState:
        return Stage1SessionState(
            format_version=self.session_format_version,
            session_id=session_id,
            memory_version=0,
            model_signature=self.model_signature(),
            memory_state=self.memory.init_state(1, self.backbone.device),
            stats=Stage1SessionStats(),
        )

    def clone_session_state(self, state: Stage1SessionState) -> Stage1SessionState:
        return Stage1SessionState(
            format_version=state.format_version,
            session_id=state.session_id,
            memory_version=state.memory_version,
            model_signature=dict(state.model_signature),
            memory_state=state.memory_state.detach().clone(),
            stats=Stage1SessionStats(**asdict(state.stats)),
        )

    def serialize_session_state(self, state: Stage1SessionState) -> dict[str, Any]:
        return {
            "format_version": state.format_version,
            "session_id": state.session_id,
            "memory_version": state.memory_version,
            "model_signature": dict(state.model_signature),
            "memory_state": state.memory_state.detach().cpu(),
            "stats": asdict(state.stats),
        }

    def deserialize_session_state(self, payload: dict[str, Any]) -> Stage1SessionState:
        state = Stage1SessionState(
            format_version=str(payload["format_version"]),
            session_id=str(payload["session_id"]),
            memory_version=int(payload["memory_version"]),
            model_signature=dict(payload["model_signature"]),
            memory_state=payload["memory_state"].to(self.backbone.device),
            stats=Stage1SessionStats(**dict(payload["stats"])),
        )
        self.validate_session_state(state)
        return state

    def validate_session_state(self, state: Stage1SessionState) -> None:
        if state.format_version != self.session_format_version:
            raise ValueError(
                f"Unexpected session format {state.format_version}, expected {self.session_format_version}"
            )
        if state.model_signature != self.model_signature():
            raise ValueError("Session state model signature does not match runtime model")
        expected_shape = (1, self.config.memory_slots, self.backbone.hidden_size)
        if tuple(state.memory_state.shape) != expected_shape:
            raise ValueError(
                f"Unexpected memory state shape {tuple(state.memory_state.shape)}, expected {expected_shape}"
            )

    def write_texts(
        self,
        state: Stage1SessionState,
        texts: list[str],
        *,
        updated_at: float | None = None,
    ) -> tuple[Stage1SessionState, dict[str, float | int]]:
        if not texts:
            return self.clone_session_state(state), {
                "num_texts": 0,
                "tokenize_s": 0.0,
                "encode_s": 0.0,
                "update_s": 0.0,
                "total_s": 0.0,
            }
        self.validate_session_state(state)
        total_started_at = time.perf_counter()
        next_state = self.clone_session_state(state)
        tokenize_started_at = time.perf_counter()
        input_ids, attention_mask = self._tokenize_texts(texts)
        tokenize_elapsed = time.perf_counter() - tokenize_started_at
        encode_started_at = time.perf_counter()
        with torch.no_grad():
            hidden_states = self.backbone.encode_tokens(input_ids, attention_mask)
            write_reprs = self._pool_hidden(hidden_states, attention_mask)
        encode_elapsed = time.perf_counter() - encode_started_at
        update_started_at = time.perf_counter()
        with torch.no_grad():
            for write_repr in write_reprs.unbind(dim=0):
                next_memory, _ = self.memory.update(next_state.memory_state, write_repr.unsqueeze(0))
                next_state.memory_state = next_memory
                next_state.memory_version += 1
                next_state.stats.num_writes += 1
        update_elapsed = time.perf_counter() - update_started_at
        if updated_at is not None:
            next_state.stats.updated_at = updated_at
        return next_state, {
            "num_texts": len(texts),
            "tokenize_s": tokenize_elapsed,
            "encode_s": encode_elapsed,
            "update_s": update_elapsed,
            "total_s": time.perf_counter() - total_started_at,
        }

    def retrieve_from_state(
        self,
        state: Stage1SessionState,
        query: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.validate_session_state(state)
        formatted_query = build_stage1_question_prompt(query, self.config.prompt_version)
        input_ids, attention_mask = self._tokenize_texts([formatted_query])
        question_hidden = self.backbone.encode_tokens(input_ids, attention_mask)
        question_repr = self._pool_hidden(question_hidden, attention_mask)
        retrieved, retrieval_weights = self.memory.retrieve(state.memory_state, question_repr)
        return retrieved, retrieval_weights, input_ids, attention_mask

    def build_chat_inputs(
        self,
        state: Stage1SessionState,
        query: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        retrieved, retrieval_weights, input_ids, attention_mask = self.retrieve_from_state(
            state,
            query,
        )
        token_embeds = self.backbone.embed_input_ids(input_ids)
        prefix_embeds = retrieved.unsqueeze(1).to(dtype=token_embeds.dtype)
        prefix_attention_mask = torch.ones(
            (1, 1),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        full_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        full_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        return full_embeds, full_attention_mask, retrieval_weights

    def answer_query(
        self,
        state: Stage1SessionState,
        query: str,
        generation_config: dict[str, Any] | None = None,
        *,
        updated_at: float | None = None,
    ) -> dict[str, Any]:
        total_started_at = time.perf_counter()
        session_snapshot = self.clone_session_state(state)
        retrieve_started_at = time.perf_counter()
        full_embeds, full_attention_mask, retrieval_weights = self.build_chat_inputs(
            session_snapshot,
            query,
        )
        retrieve_elapsed = time.perf_counter() - retrieve_started_at
        generate_started_at = time.perf_counter()
        output_ids = self.backbone.generate_from_embeds(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            generation_config=generation_config,
        )
        generate_elapsed = time.perf_counter() - generate_started_at
        decode_started_at = time.perf_counter()
        answer = self.backbone.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        decode_elapsed = time.perf_counter() - decode_started_at
        if updated_at is not None:
            session_snapshot.stats.updated_at = updated_at
        session_snapshot.stats.num_queries += 1
        return {
            "answer": answer,
            "memory_version": session_snapshot.memory_version,
            "retrieval_weights": retrieval_weights.detach().cpu(),
            "session_state": session_snapshot,
            "profile": {
                "path": "memory",
                "retrieve_s": retrieve_elapsed,
                "generate_s": generate_elapsed,
                "decode_s": decode_elapsed,
                "total_s": time.perf_counter() - total_started_at,
            },
        }

    def answer_query_direct(
        self,
        query: str,
        generation_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        total_started_at = time.perf_counter()
        formatted_query = build_stage1_question_prompt(query, self.config.prompt_version)
        tokenize_started_at = time.perf_counter()
        input_ids, attention_mask = self._tokenize_texts([formatted_query])
        tokenize_elapsed = time.perf_counter() - tokenize_started_at
        generate_started_at = time.perf_counter()
        output_ids = self.backbone.generate_from_input_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
        generate_elapsed = time.perf_counter() - generate_started_at
        decode_started_at = time.perf_counter()
        answer = self.backbone.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        decode_elapsed = time.perf_counter() - decode_started_at
        return {
            "answer": answer,
            "retrieval_weights": None,
            "profile": {
                "path": "direct_backbone",
                "tokenize_s": tokenize_elapsed,
                "generate_s": generate_elapsed,
                "decode_s": decode_elapsed,
                "total_s": time.perf_counter() - total_started_at,
            },
        }

    def load_trainable_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = False) -> None:
        own_state = self.state_dict()
        filtered = {name: tensor for name, tensor in state_dict.items() if name in own_state}
        self.load_state_dict(filtered, strict=strict)

    def forward(
        self,
        history_input_ids: torch.Tensor,
        history_attention_mask: torch.Tensor,
        history_chunk_mask: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        write_counts: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        batch_size, max_history_chunks, _ = history_input_ids.shape
        device = input_ids.device
        memory_state = self.memory.init_state(batch_size, device)
        write_gate_logits: list[torch.Tensor] = []

        for chunk_index in range(max_history_chunks):
            active_samples = history_chunk_mask[:, chunk_index]
            if not bool(active_samples.any()):
                continue
            chunk_input_ids = history_input_ids[:, chunk_index]
            chunk_attention_mask = history_attention_mask[:, chunk_index]
            hidden_states = self.backbone.encode_tokens(chunk_input_ids, chunk_attention_mask)
            write_repr = self._pool_hidden(hidden_states, chunk_attention_mask)
            if self.config.use_write_gate_loss:
                predicted_gate_logits = self.write_decision_head(write_repr).squeeze(-1)
                write_gate_logits.append(predicted_gate_logits)
            memory_state, _ = self.memory.update(memory_state, write_repr)

        question_hidden = self.backbone.encode_tokens(question_input_ids, question_attention_mask)
        question_repr = self._pool_hidden(question_hidden, question_attention_mask)
        retrieved, retrieval_weights = self.memory.retrieve(memory_state, question_repr)

        token_embeds = self.backbone.embed_input_ids(input_ids)
        prefix_embeds = retrieved.unsqueeze(1).to(dtype=token_embeds.dtype)
        prefix_attention_mask = torch.ones(
            (batch_size, 1),
            device=device,
            dtype=attention_mask.dtype,
        )
        full_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        full_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        prefix_labels = torch.full(
            (batch_size, 1),
            fill_value=-100,
            device=device,
            dtype=labels.dtype,
        )
        full_labels = torch.cat([prefix_labels, labels], dim=1)
        outputs = self.backbone.forward_with_embeds(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )
        if outputs.loss is None:
            raise RuntimeError("Backbone did not return loss")

        write_gate_loss = torch.tensor(0.0, device=device)
        if self.config.use_write_gate_loss and write_counts is not None and write_gate_logits:
            average_gate_logits = torch.stack(write_gate_logits, dim=1).mean(dim=1)
            write_targets = (write_counts > 0).to(dtype=average_gate_logits.dtype)
            write_gate_loss = nn.functional.binary_cross_entropy_with_logits(
                average_gate_logits,
                write_targets,
            )

        total_loss = outputs.loss + 0.1 * write_gate_loss
        return {
            "loss": total_loss,
            "answer_loss": outputs.loss.detach(),
            "write_gate_loss": write_gate_loss.detach(),
            "logits": outputs.logits[:, 1:, :],
            "retrieval_weights": retrieval_weights.detach(),
        }


def build_stage1_model(config: Stage1ModelConfig) -> FrozenBackboneWithTimelineMemory:
    return FrozenBackboneWithTimelineMemory(config)
