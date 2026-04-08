from __future__ import annotations

import hashlib
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None

from titans.neural_memory import NLMConfig, NLMState, NeuralLongTermMemory
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
    memory_slots: int = 16  # kept for backward compat; not used by NLM
    memory_hidden_mult: float = 4.0
    memory_dropout: float = 0.0
    trust_remote_code: bool = False
    prompt_version: str = DEFAULT_STAGE1_PROMPT_VERSION
    use_write_gate_loss: bool = False
    # NLM config
    num_memory_layers: int = 1
    num_memory_tokens: int = 64
    memory_lr: float = 0.1
    memory_momentum: float = 0.9
    memory_decay: float = 0.01
    nlm_init_std: float = 0.02
    # LoRA config
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target: str = "q_proj,k_proj,v_proj,o_proj"
    lora_dropout: float = 0.05


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
    memory_state: NLMState
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

        self.use_lora = config.use_lora
        if config.use_lora:
            self._apply_lora(config)
        else:
            self.freeze_parameters()

    def _apply_lora(self, config: Stage1ModelConfig) -> None:
        """Apply LoRA to backbone attention layers."""
        from peft import LoraConfig, TaskType, get_peft_model

        target_modules = [m.strip() for m in config.lora_target.split(",")]
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        # Freeze non-LoRA parameters
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad_(False)

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
        if self.use_lora:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
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
        ctx = self.model.disable_adapter() if self.use_lora else nullcontext()
        with torch.inference_mode(), ctx:
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **config,
            )


class FrozenBackboneWithTimelineMemory(nn.Module):
    session_format_version = "stage1-session-v2"

    def __init__(self, config: Stage1ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = FrozenBackboneAdapter(config)

        nlm_config = NLMConfig(
            dim=self.backbone.hidden_size,
            num_memory_layers=config.num_memory_layers,
            memory_hidden_mult=config.memory_hidden_mult,
            memory_lr=config.memory_lr,
            memory_momentum=config.memory_momentum,
            memory_decay=config.memory_decay,
            init_std=config.nlm_init_std,
        )
        self.memory = NeuralLongTermMemory(nlm_config)

        # Learnable query tokens for memory retrieval
        self.memory_query_tokens = nn.Parameter(
            torch.randn(config.num_memory_tokens, self.backbone.hidden_size) * 0.02
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
            "num_memory_layers": self.config.num_memory_layers,
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
            memory_state=self.memory.init_state(self.backbone.device),
            stats=Stage1SessionStats(),
        )

    def clone_session_state(self, state: Stage1SessionState) -> Stage1SessionState:
        return Stage1SessionState(
            format_version=state.format_version,
            session_id=state.session_id,
            memory_version=state.memory_version,
            model_signature=dict(state.model_signature),
            memory_state=state.memory_state.clone(),
            stats=Stage1SessionStats(**asdict(state.stats)),
        )

    def serialize_session_state(self, state: Stage1SessionState) -> dict[str, Any]:
        return {
            "format_version": state.format_version,
            "session_id": state.session_id,
            "memory_version": state.memory_version,
            "model_signature": dict(state.model_signature),
            "memory_weights": [w.detach().cpu() for w in state.memory_state.weights],
            "memory_momentum": [m.detach().cpu() for m in state.memory_state.momentum],
            "stats": asdict(state.stats),
        }

    def deserialize_session_state(self, payload: dict[str, Any]) -> Stage1SessionState:
        device = self.backbone.device
        memory_state = NLMState(
            weights=[w.to(device) for w in payload["memory_weights"]],
            momentum=[m.to(device) for m in payload["memory_momentum"]],
        )
        state = Stage1SessionState(
            format_version=str(payload["format_version"]),
            session_id=str(payload["session_id"]),
            memory_version=int(payload["memory_version"]),
            model_signature=dict(payload["model_signature"]),
            memory_state=memory_state,
            stats=Stage1SessionStats(**dict(payload["stats"])),
        )
        return state

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
        total_started_at = time.perf_counter()
        next_state = self.clone_session_state(state)
        tokenize_started_at = time.perf_counter()
        input_ids, attention_mask = self._tokenize_texts(texts)
        tokenize_elapsed = time.perf_counter() - tokenize_started_at
        encode_started_at = time.perf_counter()
        with torch.no_grad():
            hidden_states = self.backbone.encode_tokens(input_ids, attention_mask)
        encode_elapsed = time.perf_counter() - encode_started_at
        update_started_at = time.perf_counter()
        # NLM write: pass hidden_states through NLM's write method
        # hidden_states shape: (batch, seq, dim) — we use it directly
        new_memory = self.memory.write(hidden_states, next_state.memory_state)
        next_state.memory_state = new_memory
        next_state.memory_version += len(texts)
        next_state.stats.num_writes += len(texts)
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
    ) -> torch.Tensor:
        """Retrieve from memory using learnable query tokens.

        Returns:
            retrieved: (1, num_memory_tokens, dim)
        """
        # Use learnable query tokens, not question hidden states
        query = self.memory_query_tokens.unsqueeze(0)  # (1, 64, dim)
        retrieved = self.memory.retrieve(query, state.memory_state)
        return retrieved

    def build_chat_inputs(
        self,
        state: Stage1SessionState,
        query: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        retrieved = self.retrieve_from_state(state)  # (1, 64, dim)
        formatted_query = build_stage1_question_prompt(query, self.config.prompt_version)
        input_ids, attention_mask = self._tokenize_texts([formatted_query])
        token_embeds = self.backbone.embed_input_ids(input_ids)
        # Scale prefix to match token embedding norm
        prefix = retrieved.to(dtype=token_embeds.dtype)
        token_norm = token_embeds.norm(dim=-1, keepdim=True).mean()
        prefix_norm = prefix.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        prefix = prefix * (token_norm / prefix_norm)
        prefix_attention_mask = torch.ones(
            (1, prefix.shape[1]),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        full_embeds = torch.cat([prefix, token_embeds], dim=1)
        full_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        return full_embeds, full_attention_mask

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
        full_embeds, full_attention_mask = self.build_chat_inputs(session_snapshot, query)
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
            "retrieval_weights": None,
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

    def load_trainable_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        own_state = self.state_dict()
        trainable_names = {name for name, p in self.named_parameters() if p.requires_grad}

        # checkpoint 中有但模型中没有的 key → LoRA checkpoint 加载到非 LoRA 模型
        missing_from_model = [k for k in state_dict if k not in own_state]
        # 模型中可训练但 checkpoint 中没有的 key → 非 LoRA checkpoint 加载到 LoRA 模型
        missing_from_ckpt = [k for k in trainable_names if k not in state_dict]

        if missing_from_model:
            raise RuntimeError(
                f"Checkpoint contains {len(missing_from_model)} keys not in model "
                f"(config mismatch?): {missing_from_model[:5]}"
            )
        if missing_from_ckpt:
            raise RuntimeError(
                f"Model has {len(missing_from_ckpt)} trainable keys not in checkpoint "
                f"(config mismatch?): {missing_from_ckpt[:5]}"
            )

        filtered = {name: tensor for name, tensor in state_dict.items() if name in own_state}
        self.load_state_dict(filtered, strict=False)

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

        # Initialize NLM state
        nlm_state = self.memory.init_state(device)

        for chunk_index in range(max_history_chunks):
            active_samples = history_chunk_mask[:, chunk_index]
            if not bool(active_samples.any()):
                continue
            chunk_input_ids = history_input_ids[:, chunk_index]
            chunk_attention_mask = history_attention_mask[:, chunk_index]
            hidden_states = self.backbone.encode_tokens(chunk_input_ids, chunk_attention_mask)
            # Write hidden_states into NLM
            nlm_state = self.memory.write(hidden_states, nlm_state)

        # Retrieve using learnable query tokens
        query_tokens = self.memory_query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        retrieved = self.memory.retrieve(query_tokens, nlm_state)  # (batch, 64, dim)

        # Build input with memory prefix (scale to match token embedding norm)
        token_embeds = self.backbone.embed_input_ids(input_ids)
        prefix = retrieved.to(dtype=token_embeds.dtype)
        token_norm = token_embeds.norm(dim=-1, keepdim=True).mean()
        prefix_norm = prefix.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        prefix = prefix * (token_norm / prefix_norm)
        prefix_attention_mask = torch.ones(
            (batch_size, prefix.shape[1]),
            device=device,
            dtype=attention_mask.dtype,
        )
        full_embeds = torch.cat([prefix, token_embeds], dim=1)
        full_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        prefix_labels = torch.full(
            (batch_size, prefix.shape[1]),
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

        total_loss = outputs.loss
        return {
            "loss": total_loss,
            "answer_loss": outputs.loss.detach(),
            "write_gate_loss": torch.tensor(0.0, device=device),
            "logits": outputs.logits[:, 1:, :],
            "retrieval_weights": torch.zeros(1, device=device),
        }


def build_stage1_model(config: Stage1ModelConfig) -> FrozenBackboneWithTimelineMemory:
    return FrozenBackboneWithTimelineMemory(config)
