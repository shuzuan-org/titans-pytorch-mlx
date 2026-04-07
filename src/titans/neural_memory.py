"""
Neural Long-term Memory for Titans MAC (adapted from ../titans/src/titans/memory.py).

Simplified for stage1 usage:
- No conv (use_conv=False)
- No CUDA optimization dependency
- Single-layer linear memory by default
- Compatible with FrozenBackboneWithTimelineMemory's write/retrieve pattern

Key equations (Titans paper):
    Memory update: M_t = (1 - alpha_t) * M_{t-1} + S_t
    Surprise: S_t = eta_t * S_{t-1} - theta_t * grad(loss(M_{t-1}; x_t))
    Loss: loss(M; x) = ||M(k) - v||^2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class NLMConfig:
    """Configuration for NeuralLongTermMemory."""

    dim: int = 3584
    num_memory_layers: int = 1
    memory_hidden_mult: float = 4.0
    memory_lr: float = 0.1
    memory_momentum: float = 0.9
    memory_decay: float = 0.01
    init_std: float = 0.02
    activation: str = "silu"

    @property
    def memory_hidden_dim(self) -> int:
        return int(self.dim * self.memory_hidden_mult)


@dataclass
class NLMState:
    """State of the neural long-term memory.

    Attributes:
        weights: List of weight matrices for each memory layer
        momentum: Accumulated surprise momentum (S_t in paper)
    """

    weights: list[torch.Tensor]
    momentum: list[torch.Tensor]

    def detach(self) -> NLMState:
        return NLMState(
            weights=[w.detach() for w in self.weights],
            momentum=[m.detach() for m in self.momentum],
        )

    def clone(self) -> NLMState:
        return NLMState(
            weights=[w.clone() for w in self.weights],
            momentum=[m.clone() for m in self.momentum],
        )

    def to(self, device: torch.device) -> NLMState:
        return NLMState(
            weights=[w.to(device) for w in self.weights],
            momentum=[m.to(device) for m in self.momentum],
        )


def _get_activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")


class MemoryMLP(nn.Module):
    """MLP that stores information in its weights.

    For num_layers=1: single Linear(dim, dim) — matrix-valued memory.
    For num_layers>=2: deep MLP with hidden layers.
    """

    def __init__(self, config: NLMConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        num_layers = config.num_memory_layers
        dim = config.dim
        hidden_dim = config.memory_hidden_dim

        if num_layers == 1:
            self.layers.append(nn.Linear(dim, dim, bias=False))
        else:
            self.layers.append(nn.Linear(dim, hidden_dim, bias=False))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.layers.append(nn.Linear(hidden_dim, dim, bias=False))

        self.activation = _get_activation(config.activation)
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=config.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < last:
                h = self.activation(h)
        return h

    def get_weights(self) -> list[torch.Tensor]:
        return [layer.weight.data.clone() for layer in self.layers]

    def set_weights(self, weights: list[torch.Tensor]) -> None:
        for layer, w in zip(self.layers, weights):
            layer.weight.data.copy_(w)


class NeuralLongTermMemory(nn.Module):
    """Neural Long-term Memory Module (Titans).

    Learns to memorize at test time using gradient descent with
    momentum and weight decay.
    """

    def __init__(self, config: NLMConfig) -> None:
        super().__init__()
        self.config = config

        self.memory = MemoryMLP(config)
        # MemoryMLP weights are memory state, managed by write().
        # Frozen — not updated by optimizer.
        self.memory.requires_grad_(False)

        # Write-path: proj_k, proj_v, gates.
        # These ARE trainable — gradients flow back through the differentiable
        # write formula (closed-form for single-layer linear memory).
        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)

        self.gate_decay = nn.Sequential(nn.Linear(config.dim, config.dim), nn.Sigmoid())
        self.gate_lr = nn.Sequential(nn.Linear(config.dim, config.dim), nn.Sigmoid())
        self.gate_momentum = nn.Sequential(nn.Linear(config.dim, config.dim), nn.Sigmoid())

        # Retrieve-path: proj_q, proj_out. Also trainable.
        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)

        for module in [self.proj_k, self.proj_v, self.proj_q, self.proj_out]:
            nn.init.normal_(module.weight, std=config.init_std)

    def init_state(self, device: torch.device | None = None) -> NLMState:
        """Initialize empty memory state."""
        if device is None:
            device = next(self.memory.parameters()).device
        weights = [
            torch.empty_like(l.weight.data, device=device).normal_(std=0.01)
            for l in self.memory.layers
        ]
        momentum = [torch.zeros_like(l.weight.data, device=device) for l in self.memory.layers]
        return NLMState(weights=weights, momentum=momentum)

    def _compute_gradients(self, keys: torch.Tensor, values: torch.Tensor) -> list[torch.Tensor]:
        """Compute gradients of associative memory loss w.r.t. memory weights."""
        if len(self.memory.layers) == 1:
            k = keys.detach().reshape(-1, keys.shape[-1])
            v = values.detach().reshape(-1, values.shape[-1])
            W = self.memory.layers[0].weight.data
            error = k @ W.T - v
            grad_W = (2.0 / error.shape[-1]) * error.T @ k
            return [grad_W]

        with torch.enable_grad():
            try:
                for param in self.memory.parameters():
                    param.requires_grad_(True)
                loss = self.memory.compute_loss(keys.detach(), values.detach())
                grads = torch.autograd.grad(
                    loss, list(self.memory.parameters()),
                    create_graph=False, allow_unused=True,
                )
            finally:
                for param in self.memory.parameters():
                    param.requires_grad_(False)
        return [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, self.memory.parameters())
        ]

    @staticmethod
    def _broadcast_gate(gate: torch.Tensor | float, w: torch.Tensor) -> torch.Tensor | float:
        if not isinstance(gate, torch.Tensor) or gate.dim() == 0:
            return gate
        if gate.shape[0] == w.shape[0]:
            return gate.unsqueeze(-1)
        return gate.mean()

    def write(self, x: torch.Tensor, state: NLMState) -> NLMState:
        """Write new information into memory (differentiable, single-layer only).

        Instead of per-token loop, computes a single aggregated update:
            k_all, v_all = proj(x)  over all tokens
            error_all = k_all @ W.T - v_all
            grad_W = (2/D) * mean(error).T @ mean(k)
            momentum_new = eta * momentum_old - theta * grad_W
            W_new = (1 - alpha) * W_old + momentum_new

        This avoids storing T intermediate computation graphs.
        """
        assert len(self.memory.layers) == 1, "Differentiable write only supports num_memory_layers=1"

        target_dtype = self.proj_k.weight.dtype
        x = x.to(dtype=target_dtype)

        k = F.normalize(F.silu(self.proj_k(x)), p=2, dim=-1)  # (batch, seq, dim)
        v = F.silu(self.proj_v(x))  # (batch, seq, dim)

        # Average gates over all tokens and batch → single set of scalars
        alpha = (self.gate_decay(x) * self.config.memory_decay).mean(dim=(0, 1))  # (dim,)
        theta = (self.gate_lr(x) * self.config.memory_lr).mean(dim=(0, 1))
        eta = (self.gate_momentum(x) * self.config.memory_momentum).mean(dim=(0, 1))

        W = state.weights[0]  # (dim_out, dim_in)
        momentum = state.momentum[0]

        # Aggregate over batch and seq: mean of k and mean of error
        k_mean = k.mean(dim=(0, 1))  # (dim,)
        v_mean = v.mean(dim=(0, 1))  # (dim,)
        error_mean = k_mean @ W.T - v_mean  # (dim_out,)

        # Closed-form gradient (aggregated)
        grad_W = (2.0 / error_mean.shape[-1]) * error_mean.unsqueeze(1) * k_mean.unsqueeze(0)  # (dim_out, dim_in)

        # Broadcast gates
        alpha_b = alpha.unsqueeze(-1)  # (dim, 1)
        theta_b = theta.unsqueeze(-1)
        eta_b = eta.unsqueeze(-1)

        # Single update step
        momentum = eta_b * momentum - theta_b * grad_W
        W = (1 - alpha_b) * W + momentum

        return NLMState(weights=[W], momentum=[momentum])

    def retrieve(self, queries: torch.Tensor, state: NLMState) -> torch.Tensor:
        """Retrieve from memory without updating.

        For single-layer linear memory: output = proj_out(W @ proj_q(queries))
        Uses state.weights directly (not set_weights) to preserve gradient chain.
        """
        assert len(self.memory.layers) == 1, "Differentiable retrieve only supports num_memory_layers=1"
        target_dtype = self.proj_q.weight.dtype
        queries = queries.to(dtype=target_dtype)
        q = F.normalize(F.silu(self.proj_q(queries)), p=2, dim=-1)
        # Direct matrix multiply with state weights (preserves gradient)
        W = state.weights[0]  # (dim_out, dim_in)
        retrieved = q @ W.T   # (batch, seq, dim_out)
        return self.proj_out(retrieved)

    def compute_loss(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute associative memory loss: ||M(k) - v||^2"""
        predictions = self.memory(keys)
        return ((predictions - values) ** 2).mean(dim=-1).sum()
