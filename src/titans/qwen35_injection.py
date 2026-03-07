# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
NLM injection for Qwen3.5 (Memory Oracle route).

Works with any Qwen3.x model that exposes model.model.layers.
Each decoder layer is wrapped with a NeuralLongTermMemory (MAL style):
    x → Memory → x + mem_out → Attention → FFN

Key difference from tptt.py:
    - Supports freeze_memory_updates() / unfreeze_memory_updates() to put
      the NLM in read-only (retrieval-only) mode without weight updates.
      This is used by MemoryOracle.read() to query memory without changing it.
    - Class names are distinct so models with both Qwen2.5 (tptt.py) and
      Qwen3.5 (this file) injections can coexist in the same process.

Public API:
    Qwen35LayerWithMemory        - layer wrapper
    inject_memory_into_qwen35    - replace decoder layers in-place
    reset_memory_states          - call before each new document / batch
    freeze_memory_updates        - enter read-only retrieval mode
    unfreeze_memory_updates      - exit read-only retrieval mode
    get_trainable_params         - set requires_grad; return trainable list
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn

from titans.config import TitansConfig
from titans.memory import MemoryState, NeuralLongTermMemory

logger = logging.getLogger(__name__)


class Qwen35LayerWithMemory(nn.Module):
    """Qwen3DecoderLayer wrapped with a NeuralLongTermMemory (MAL style).

    Forward order:
        hidden_states → NeuralMemory → residual add → original Qwen layer

    Two operating modes:
        Normal (frozen=False): NLM retrieves AND updates memory state.
        Frozen (frozen=True):  NLM retrieves ONLY; memory_state is unchanged.
            Used by MemoryOracle.read() to prevent query contamination.

    Memory state is stored as an instance attribute; reset before each batch
    via reset_memory_states(model).
    """

    def __init__(
        self,
        original_layer: nn.Module,
        memory: NeuralLongTermMemory,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.original_layer = original_layer
        self.memory = memory
        self.layer_idx: int = layer_idx  # stable identity for state save/load
        # State for this layer; reset per-sequence
        self.memory_state: MemoryState | None = None
        # When True: retrieval only, no weight update
        self._nlm_frozen: bool = False

    # ------------------------------------------------------------------
    # Attribute delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        # Delegate unknown attributes to the wrapped layer so that
        # transformers internals (e.g. attention_type, layer_idx) still work.
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules", {})
            original = modules.get("original_layer")
            # Guard against double-wrapping: if original is also a
            # Qwen35LayerWithMemory its __getattr__ would recurse infinitely.
            if original is not None and not isinstance(original, Qwen35LayerWithMemory):
                return getattr(original, name)
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ):
        """MAL forward: memory first, then the original Qwen layer.

        Args:
            hidden_states: (batch, seq, dim)
            **kwargs: Forwarded unchanged to the original decoder layer.

        Returns:
            Whatever the original Qwen decoder layer returns.
        """
        if self._nlm_frozen:
            # Read-only: retrieve without updating state
            if self.memory_state is None:
                logger.warning(
                    "Layer %d: retrieve called in frozen mode with no prior write "
                    "— returning near-zero output from empty memory",
                    self.layer_idx,
                )
                self.memory_state = self.memory.init_state()
            # retrieve() applies proj_q + memory.forward() + proj_out
            mem_out = self.memory.retrieve(hidden_states, self.memory_state)
        else:
            # Normal: retrieve AND update state
            mem_out, self.memory_state = self.memory(
                hidden_states,
                state=self.memory_state,
                update_memory=True,
            )

        hidden_states = hidden_states + mem_out
        return self.original_layer(hidden_states, **kwargs)


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


def _get_text_backbone(model: nn.Module) -> tuple[nn.Module, int]:
    """Return (text_backbone, hidden_size) for Qwen3/Qwen2.5/Qwen3.5.

    Qwen3.5ForConditionalGeneration (multimodal):
        text_backbone = model.language_model
        hidden_size   = model.config.text_config.hidden_size

    Qwen3ForCausalLM / Qwen2.5ForCausalLM (text-only):
        text_backbone = model
        hidden_size   = model.config.hidden_size
    """
    if hasattr(model, "language_model"):
        # Multimodal: Qwen3.5ForConditionalGeneration
        text_backbone = model.language_model
        hidden_size = model.config.text_config.hidden_size
    else:
        # Text-only: Qwen3/Qwen2.5 CausalLM
        text_backbone = model
        hidden_size = model.config.hidden_size
    return text_backbone, hidden_size


def inject_memory_into_qwen35(
    model: nn.Module,
    mem_cfg: TitansConfig,
) -> nn.Module:
    """Replace every decoder layer with a Qwen35LayerWithMemory.

    Each layer gets its own independent NeuralLongTermMemory on the same
    device / dtype as the base model.

    Supports both:
    - Qwen3ForCausalLM / Qwen2.5ForCausalLM (text-only, model.model.layers)
    - Qwen3.5ForConditionalGeneration (multimodal, model.language_model.model.layers)

    Args:
        model: Qwen model with model.model.layers or language_model.model.layers.
        mem_cfg: TitansConfig — dim must match the model's hidden_size.

    Returns:
        The mutated model (same object, layers replaced in-place).
    """
    text_backbone, hidden_size = _get_text_backbone(model)

    if mem_cfg.dim != hidden_size:
        raise ValueError(
            f"mem_cfg.dim ({mem_cfg.dim}) != model hidden_size ({hidden_size})"
        )

    ref_param = next(model.parameters())
    dtype = ref_param.dtype
    device = ref_param.device

    original_layers = text_backbone.model.layers
    new_layers: list[nn.Module] = []

    for idx, layer in enumerate(original_layers):
        memory = NeuralLongTermMemory(mem_cfg).to(dtype=dtype, device=device)
        new_layers.append(Qwen35LayerWithMemory(layer, memory, layer_idx=idx))

    text_backbone.model.layers = nn.ModuleList(new_layers)
    return model


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def reset_memory_states(model: nn.Module) -> None:
    """Set all Qwen35LayerWithMemory.memory_state to None.

    Call before every new document / training batch to prevent state leaking
    across independent sequences.
    """
    for module in model.modules():
        if isinstance(module, Qwen35LayerWithMemory):
            module.memory_state = None


def freeze_memory_updates(model: nn.Module) -> None:
    """Switch all NLM layers to read-only (retrieval-only) mode.

    In this mode the NLM retrieves information from the current memory state
    but does NOT update the weights.  Call before model.generate() in
    MemoryOracle.read() to prevent generation tokens from writing into memory.
    """
    for module in model.modules():
        if isinstance(module, Qwen35LayerWithMemory):
            module._nlm_frozen = True


def unfreeze_memory_updates(model: nn.Module) -> None:
    """Switch all NLM layers back to normal (write) mode."""
    for module in model.modules():
        if isinstance(module, Qwen35LayerWithMemory):
            module._nlm_frozen = False


# ---------------------------------------------------------------------------
# Trainable parameters
# ---------------------------------------------------------------------------


def get_trainable_params(model: nn.Module) -> list[nn.Parameter]:
    """Set requires_grad and return trainable parameters.

    Trainable:
        - NLM structural params: proj_k/v/q/out, gate_decay/lr/momentum,
          conv_k/v/q (if used).  These are identified by isinstance traversal
          so they survive peft wrapper renaming.
        - LoRA adapter params: identified by 'lora_' in parameter name.

    Frozen:
        - Base model weights.
        - MemoryMLP layer weights (updated by the Titans mechanism, not AdamW).

    Args:
        model: Optionally after peft.get_peft_model() has been applied.

    Returns:
        List of nn.Parameter objects with requires_grad=True.
    """
    # Step 1: freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    # Step 2: unfreeze NLM structural params by direct access.
    # Cleaner than collecting id()s in one loop and cross-referencing in
    # another — mutate the param object the moment we find it.
    for module in model.modules():
        if isinstance(module, Qwen35LayerWithMemory):
            for name, param in module.memory.named_parameters():
                # Exclude MemoryMLP layer weights (updated by Titans, not AdamW)
                if not name.startswith("memory.layers."):
                    param.requires_grad_(True)

    # Step 3: unfreeze LoRA params by name
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)

    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# NLM state snapshot helpers (for MemoryOracle.save/load)
# ---------------------------------------------------------------------------


def get_nlm_states(model: nn.Module) -> dict[int, MemoryState | None]:
    """Return a dict mapping layer_idx → MemoryState (or None).

    Keys are the stable layer_idx assigned at injection time, not the
    model.modules() traversal order (which can change with peft wrapping).
    """
    states: dict[int, MemoryState | None] = {}
    for module in model.modules():
        if isinstance(module, Qwen35LayerWithMemory):
            state = module.memory_state
            states[module.layer_idx] = state.clone() if state is not None else None
    return states


def set_nlm_states(model: nn.Module, states: dict[int, MemoryState | None]) -> None:
    """Restore NLM states previously captured with get_nlm_states().

    Tensors are moved to the device of the corresponding NLM module.
    Keys in `states` must be the layer_idx values produced by get_nlm_states().
    Warns if the number of layers in the model differs from len(states).
    """
    n_restored = 0
    for module in model.modules():
        if isinstance(module, Qwen35LayerWithMemory):
            state = states.get(module.layer_idx)
            if state is None:
                module.memory_state = None
            else:
                device = next(module.memory.parameters()).device
                module.memory_state = MemoryState(
                    weights=[w.to(device) for w in state.weights],
                    momentum=[m.to(device) for m in state.momentum],
                )
            n_restored += 1

    if n_restored != len(states):
        logger.warning(
            "set_nlm_states: model has %d NLM layers but states dict has %d entries "
            "— layer count mismatch, some states may be silently dropped",
            n_restored, len(states),
        )
