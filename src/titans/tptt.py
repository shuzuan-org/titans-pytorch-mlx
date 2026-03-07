# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
TPTT (Test-time Pre-Train and Test) injection for Qwen2.5.

Wraps each Qwen2DecoderLayer with a NeuralLongTermMemory module (MAL style):
    x → Memory → x + mem_out → Attention → FFN

This file provides:
    QwenLayerWithMemory  - single layer wrapper
    inject_memory_into_qwen  - replace all decoder layers in-place
    reset_memory_states  - reset stateful memory (call before each batch)
    get_trainable_params - enable grad for memory + LoRA, freeze the rest
"""

from __future__ import annotations

import torch
import torch.nn as nn

from titans.config import TitansConfig
from titans.memory import MemoryState, NeuralLongTermMemory


class QwenLayerWithMemory(nn.Module):
    """Qwen2DecoderLayer wrapped with a NeuralLongTermMemory (MAL style).

    Forward order (MAL):
        hidden_states → NeuralMemory → residual add → original Qwen layer

    Memory state is stored as an instance attribute and is reset by calling
    reset_memory_states(model). This keeps the API identical to vanilla layers:
    callers pass the same kwargs they would to Qwen2DecoderLayer.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        memory: NeuralLongTermMemory,
    ) -> None:
        super().__init__()
        self.original_layer = original_layer
        self.memory = memory
        # Not a parameter/buffer; reset before each batch
        self.memory_state: MemoryState | None = None

    def __getattr__(self, name: str):
        # nn.Module.__getattr__ handles parameters/buffers/submodules.
        # For anything else (e.g. attention_type, layer_idx that transformers
        # internals may access on decoder layers), delegate to original_layer.
        #
        # NOTE: nn.Module stores submodules in self._modules (not __dict__),
        # so we must look up original_layer via _modules, NOT via
        # object.__getattribute__, to avoid a secondary AttributeError.
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules", {})
            original = modules.get("original_layer")
            if original is not None:
                return getattr(original, name)
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ):
        """MAL forward: memory first, then the original Qwen layer.

        Args:
            hidden_states: (batch, seq, dim)
            **kwargs: Passed through to the original Qwen2DecoderLayer unchanged.

        Returns:
            Whatever Qwen2DecoderLayer.forward returns (tuple with hidden_states
            as first element, optionally followed by attn weights / kv cache).
        """
        # Memory retrieval + update; residual connection
        mem_out, self.memory_state = self.memory(
            hidden_states,
            state=self.memory_state,
            update_memory=True,
        )
        hidden_states = hidden_states + mem_out

        # Original Qwen2DecoderLayer (attention + FFN)
        return self.original_layer(hidden_states, **kwargs)


def inject_memory_into_qwen(
    model: nn.Module,
    mem_cfg: TitansConfig,
) -> nn.Module:
    """Replace every Qwen2DecoderLayer with a QwenLayerWithMemory.

    Each layer gets its own independent NeuralLongTermMemory instance placed
    on the same device / dtype as the base model.

    Args:
        model: Qwen2ForCausalLM (or compatible model with model.model.layers).
        mem_cfg: TitansConfig with dim matching the model's hidden_size.

    Returns:
        The mutated model (same object, layers replaced in-place).

    Raises:
        ValueError: if mem_cfg.dim doesn't match model.config.hidden_size.
    """
    # Validate dim before touching anything
    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
        if mem_cfg.dim != model.config.hidden_size:
            raise ValueError(
                f"mem_cfg.dim ({mem_cfg.dim}) != model.config.hidden_size "
                f"({model.config.hidden_size}) — pass TitansConfig(dim=model.config.hidden_size)"
            )

    # Infer dtype/device from existing params
    ref_param = next(model.parameters())
    dtype = ref_param.dtype
    device = ref_param.device

    original_layers = model.model.layers
    new_layers: list[nn.Module] = []

    for layer in original_layers:
        memory = NeuralLongTermMemory(mem_cfg).to(dtype=dtype, device=device)
        new_layers.append(QwenLayerWithMemory(layer, memory))

    model.model.layers = nn.ModuleList(new_layers)
    return model


def reset_memory_states(model: nn.Module) -> None:
    """Set all QwenLayerWithMemory.memory_state to None.

    Call this before every training batch and before processing a new document
    at inference time to prevent state leaking across independent sequences.
    """
    for module in model.modules():
        if isinstance(module, QwenLayerWithMemory):
            module.memory_state = None


def get_trainable_params(model: nn.Module) -> list[nn.Parameter]:
    """Set requires_grad and return the list of trainable parameters.

    Trainable:
        - NeuralLongTermMemory structural params (proj_*, gate_*, conv_*, proj_out)
          identified via isinstance(module, QwenLayerWithMemory) + module.memory,
          excluding MemoryMLP layer weights (updated by the Titans mechanism).
        - LoRA adapter params → identified by 'lora_' in name (peft naming is
          stable; using string match here is lower risk than for memory params).

    Frozen:
        - Base model weights, MemoryMLP.layers.* weights.

    The MemoryMLP layer weights (under NeuralLongTermMemory.memory.layers.*)
    are intentionally excluded from the optimizer: they are updated by the
    Titans gradient-descent-in-forward mechanism (_compute_gradients +
    set_weights), not by standard backprop.

    Args:
        model: Model, optionally after peft.get_peft_model() has been applied.

    Returns:
        List of nn.Parameter objects that require grad.
    """
    # Collect param IDs of NeuralLongTermMemory structural params via
    # isinstance traversal — immune to peft wrapper name changes.
    memory_param_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, QwenLayerWithMemory):
            for name, param in module.memory.named_parameters():
                # "memory" here is the MemoryMLP sub-module of
                # NeuralLongTermMemory; its layer weights start with
                # "memory.layers." and must be excluded.
                if not name.startswith("memory.layers."):
                    memory_param_ids.add(id(param))

    trainable: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        is_memory_struct = id(param) in memory_param_ids
        is_lora = "lora_" in name

        if is_memory_struct or is_lora:
            param.requires_grad_(True)
            trainable.append(param)
        else:
            param.requires_grad_(False)

    return trainable
