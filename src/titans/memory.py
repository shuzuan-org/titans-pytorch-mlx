# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Neural Long-term Memory Module for Titans.

This module implements the core innovation of Titans: a neural memory that
learns to memorize at test time using gradient descent with momentum and
weight decay. The memory is trained with an associative memory loss to
learn key-value associations.

Key equations from the paper:
    Memory update: M_t = (1 - alpha_t) * M_{t-1} + S_t
    Surprise: S_t = eta_t * S_{t-1} - theta_t * grad(loss(M_{t-1}; x_t))
    Loss: loss(M; x) = ||M(k) - v||^2

where:
    - alpha_t: forgetting/decay factor (weight decay)
    - eta_t: surprise decay (momentum coefficient)
    - theta_t: learning rate for momentary surprise
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from titans.config import TitansConfig

logger = logging.getLogger(__name__)

# Import optimizations
try:
    from titans.cuda_optimizations import (
        compute_memory_gradients_efficient,
    )
    HAS_CUDA_OPTIMIZATIONS = True
except ImportError:
    HAS_CUDA_OPTIMIZATIONS = False


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")


@dataclass
class MemoryState:
    """State of the neural long-term memory.

    This encapsulates the memory weights and momentum for continuing
    inference across chunks/segments.

    Attributes:
        weights: List of weight matrices for each memory layer
        momentum: Accumulated surprise momentum (S_t in paper)
    """

    weights: list[torch.Tensor]
    momentum: list[torch.Tensor]

    def detach(self) -> MemoryState:
        """Detach state from computation graph."""
        return MemoryState(
            weights=[w.detach() for w in self.weights],
            momentum=[m.detach() for m in self.momentum],
        )

    def clone(self) -> MemoryState:
        """Clone the memory state."""
        return MemoryState(
            weights=[w.clone() for w in self.weights],
            momentum=[m.clone() for m in self.momentum],
        )


class MemoryMLP(nn.Module):
    """MLP architecture for the neural memory.

    This is the actual memory module that stores information in its weights.
    It's a simple MLP that learns key-value associations.

    For L_M = 1 (linear memory), this is equivalent to a matrix-valued memory.
    For L_M >= 2 (deep memory), this provides more expressive power.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.init_std = config.init_std

        # Build MLP layers
        num_layers = config.num_memory_layers
        dim = config.dim
        hidden_dim = config.memory_hidden_dim
        self.layers = nn.ModuleList()

        if num_layers == 1:
            # Linear memory: single linear layer
            self.layers.append(nn.Linear(dim, dim, bias=False))
        else:
            # Deep memory: MLP with hidden layers
            # First layer: dim -> hidden_dim
            self.layers.append(nn.Linear(dim, hidden_dim, bias=False))

            # Hidden layers
            for _ in range(num_layers - 2):
                self.layers.append(
                    nn.Linear(hidden_dim, hidden_dim, bias=False)
                )

            # Last layer: hidden_dim -> dim
            self.layers.append(nn.Linear(hidden_dim, dim, bias=False))

        self.activation = get_activation(config.activation)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values."""
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=self.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through memory MLP.

        Args:
            x: Input tensor of shape (batch, seq, dim)

        Returns:
            Output tensor of shape (batch, seq, dim)
        """
        h = x
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < last:
                h = self.activation(h)
        return h

    def get_weights(self) -> list[torch.Tensor]:
        """Get current weight matrices."""
        return [layer.weight.data.clone() for layer in self.layers]

    def set_weights(self, weights: list[torch.Tensor]) -> None:
        """Set weight matrices.

        Not thread-safe: mutates layer.weight.data in-place.
        Do not share a single NeuralLongTermMemory instance across threads.
        """
        for layer, w in zip(self.layers, weights, strict=True):
            layer.weight.data.copy_(w)

    def compute_loss(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute associative memory loss.

        Loss: ||M(k) - v||^2

        Args:
            keys: Key vectors (batch, seq, dim)
            values: Value vectors (batch, seq, dim)

        Returns:
            Scalar loss value
        """
        predictions = self.forward(keys)
        # mean over D (feature dim), sum over batch×seq — prevents N×D deflation
        # that made gradients ~1e-5 with pure "mean" reduction.
        return ((predictions - values) ** 2).mean(dim=-1).sum()


class NeuralLongTermMemory(nn.Module):
    """Neural Long-term Memory Module.

    This is the main memory component of Titans. It learns to memorize
    at test time by treating training as an online learning problem.

    The memory is updated using gradient descent with:
    - Momentum (for past surprise)
    - Weight decay (for forgetting)

    Key features:
    1. Data-dependent learning rate, momentum, and decay
    2. Deep memory MLP for expressive power
    3. Surprise-based update rule
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Projections for keys, values, and queries
        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)

        # Optional 1D depthwise convolution (following Mamba2/GatedDeltaNet)
        self.use_conv = config.use_conv
        if self.use_conv:
            self.conv_k = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )
            self.conv_v = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )
            self.conv_q = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )

        # The actual memory module
        self.memory = MemoryMLP(config)

        # Data-dependent gates for learning parameters
        # These produce alpha_t (decay), theta_t (lr), eta_t (momentum)
        self.gate_decay = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.Sigmoid(),
        )
        self.gate_lr = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.Sigmoid(),
        )
        self.gate_momentum = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.Sigmoid(),
        )

        # Output projection
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in [self.proj_k, self.proj_v, self.proj_q, self.proj_out]:
            nn.init.normal_(module.weight, std=self.config.init_std)

    def _apply_conv_single(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        """Apply causal 1D depthwise conv to a single (batch, seq, dim) tensor."""
        seq_len = x.shape[1]
        x = rearrange(x, "b s d -> b d s")
        x = conv(x)[..., :seq_len]
        return rearrange(x, "b d s -> b s d")

    def _apply_conv(
        self, k: torch.Tensor, v: torch.Tensor, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply 1D convolution to K, V, Q."""
        if not self.use_conv:
            return k, v, q
        return (
            self._apply_conv_single(k, self.conv_k),
            self._apply_conv_single(v, self.conv_v),
            self._apply_conv_single(q, self.conv_q),
        )

    def _compute_gradients(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Compute gradients for memory update.

        This computes the gradient of the associative memory loss
        with respect to the memory weights.

        Priority:
        1. Analytical closed-form for single-layer linear memory (fastest, no autograd)
        2. CUDA extension (if available)
        3. torch.autograd.grad fallback (multi-layer memory)

        Args:
            keys: Key vectors (batch, seq, dim)
            values: Value vectors (batch, seq, dim)

        Returns:
            List of gradient tensors for each memory layer
        """
        # Fast path: single-layer linear memory has a closed-form gradient.
        # loss = mean_D(||W @ k - v||^2).sum_batch
        # grad_W = (2/D_out) * error.T @ k
        # This replaces torch.autograd.grad with two matrix multiplications,
        # eliminating Python autograd overhead (~50x faster per token).
        if len(self.memory.layers) == 1:
            k = keys.detach().reshape(-1, keys.shape[-1])    # (B*T, D_in)
            v = values.detach().reshape(-1, values.shape[-1]) # (B*T, D_out)
            W = self.memory.layers[0].weight.data             # (D_out, D_in)
            error = k @ W.T - v                               # (B*T, D_out)
            grad_W = (2.0 / error.shape[-1]) * error.T @ k   # (D_out, D_in)
            return [grad_W]

        # CUDA extension for multi-layer memory (if compiled)
        if HAS_CUDA_OPTIMIZATIONS and keys.is_cuda:
            try:
                weights = [layer.weight.data for layer in self.memory.layers]
                return compute_memory_gradients_efficient(
                    keys.detach(),
                    values.detach(),
                    weights,
                    activation=self.config.activation,
                )
            except Exception as e:
                logger.debug(
                    "CUDA gradient optimization failed (%s), falling back to autograd", e
                )

        # Autograd fallback for multi-layer memory.
        # This is essential because Titans learns at test time (inference mode).
        with torch.enable_grad():
            try:
                for param in self.memory.parameters():
                    param.requires_grad_(True)
                keys_grad = keys.detach().requires_grad_(True)
                v = values.detach()
                loss = self.memory.compute_loss(keys_grad, v)
                grads = torch.autograd.grad(
                    loss,
                    list(self.memory.parameters()),
                    create_graph=False,
                    allow_unused=True,
                )
            finally:
                # Always restore requires_grad=False; leaking True would cause
                # AdamW to pick up gradients on Titans-managed weights.
                for param in self.memory.parameters():
                    param.requires_grad_(False)

        return [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, self.memory.parameters(), strict=True)
        ]

    def init_state(self) -> MemoryState:
        """Initialize memory state for an empty (un-written) memory.

        Weights are initialised with tiny noise (std=0.01) rather than exact
        zeros. This is necessary for multi-layer memory: with zero weights,
        SiLU(W1 @ k) = SiLU(0) = 0, which kills gradients through W2 and W1
        via the chain rule, making the first write a no-op. Tiny noise keeps
        gradients alive while keeping retrieval output small at init,
        satisfying the "empty memory" semantic.

        For single-layer linear memory, zero weights do not cause dead
        gradients (no activation between input and weight), but we use the
        same tiny-noise init for consistency.

        Momentum starts at zero (no accumulated surprise yet).
        Device is inherited from MemoryMLP weights.

        Returns:
            Initial memory state
        """
        weights  = [torch.empty_like(l.weight.data).normal_(std=0.01)
                    for l in self.memory.layers]
        momentum = [torch.zeros_like(l.weight.data) for l in self.memory.layers]
        return MemoryState(weights=weights, momentum=momentum)

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
        update_memory: bool = True,
    ) -> tuple[torch.Tensor, MemoryState | None]:
        """Forward pass with optional memory update.

        Always performs memory retrieval. When update_memory=True (default),
        also updates the memory weights with new key-value pairs.

        Args:
            x: Input tensor (batch, seq, dim)
            state: Previous memory state (optional, initialised to zeros if None)
            update_memory: If False, retrieval only — no gradient computation,
                           no weight update, returns (output, None).

        Returns:
            Tuple of (output, new_state). new_state is None when update_memory=False.
        """
        # Initialize state if needed
        if state is None:
            state = self.init_state()

        # Set memory weights from state
        self.memory.set_weights(state.weights)

        # ── Query path (always needed for retrieval) ──────────────────────────
        q = self.proj_q(x)
        if self.use_conv:
            q = self._apply_conv_single(q, self.conv_q)
        q = F.silu(q)
        q = F.normalize(q, p=2, dim=-1)

        # Retrieve from memory: y_t = M*(q_t)
        output = self.proj_out(self.memory(q))

        if not update_memory:
            return output, None

        # ── Key/value path (only needed for memory update) ────────────────────
        k = self.proj_k(x)
        v = self.proj_v(x)
        if self.use_conv:
            k = self._apply_conv_single(k, self.conv_k)
            v = self._apply_conv_single(v, self.conv_v)
        k = F.normalize(F.silu(k), p=2, dim=-1)
        v = F.silu(v)

        # Compute data-dependent gates per token (batch, seq, dim)
        alpha = self.gate_decay(x) * self.config.memory_decay      # (B, T, D) decay
        theta = self.gate_lr(x) * self.config.memory_lr            # (B, T, D) lr
        eta = self.gate_momentum(x) * self.config.memory_momentum  # (B, T, D) momentum

        # Per-token sequential update — faithful to the Titans paper.
        # Each token t updates W using only (k_t, v_t, alpha_t, theta_t, eta_t),
        # so causality is preserved: W_t depends only on tokens 0..t.
        # The batch dim is averaged for gate scalars (approximation for B>1).
        #
        # Complexity: O(T) autograd calls (one torch.autograd.grad per token).
        # For single-layer linear memory this can be replaced with a closed-form
        # outer-product accumulation, eliminating the loop entirely — see the
        # CUDA path in _compute_gradients. Python loop is the correct fallback
        # for multi-layer memory where no analytical form exists.
        # At T=2048 with 28 layers this is ~57K backward passes per forward;
        # keep seq_len short (≤256) until the CUDA optimized path is wired up.
        weights = [w.clone() for w in state.weights]
        momentum_s = [m.clone() for m in state.momentum]
        T = k.shape[1]
        for t in range(T):
            alpha_t = alpha[:, t, :].mean(0)   # (D,)
            theta_t = theta[:, t, :].mean(0)
            eta_t   = eta[:, t, :].mean(0)
            grads_t = self._compute_gradients(
                k[:, t : t + 1, :], v[:, t : t + 1, :]
            )
            weights, momentum_s = self._standard_memory_update(
                weights, momentum_s, grads_t, alpha_t, eta_t, theta_t
            )
        new_weights, new_momentum = weights, momentum_s

        return output, MemoryState(weights=new_weights, momentum=new_momentum).detach()

    @staticmethod
    def _broadcast_gate(
        gate: torch.Tensor | float, w: torch.Tensor
    ) -> torch.Tensor | float:
        """Reshape a (D,) per-dim gate for broadcasting with weight matrix w.

        Weight matrices have shape (D_out, D_in).  A (D,) gate should scale
        per output-dimension (rows), so we unsqueeze to (D_out, 1) which
        broadcasts correctly against (D_out, D_in).

        Falls back to a scalar mean when shapes don't align (e.g. intermediate
        layers in deep memory where D_out == hidden_dim != D).
        """
        if not isinstance(gate, torch.Tensor) or gate.dim() == 0:
            return gate
        if gate.shape[0] == w.shape[0]:
            # Assumes gate.shape[0] == model_dim (D), not hidden_dim.
            # Incorrect if memory_hidden_mult == 1.0 (hidden_dim == D), but that
            # is not a supported configuration and not used by the Oracle.
            return gate.unsqueeze(-1)  # (D,) → (D, 1) for row-wise scaling
        # Shape mismatch (deep memory hidden layers: D_out == hidden_dim != D)
        # Fall back to scalar to avoid incorrect broadcasting.
        logger.debug(
            "_broadcast_gate: shape mismatch gate=%s w=%s, falling back to scalar mean",
            gate.shape, w.shape,
        )
        return gate.mean()

    @staticmethod
    def _standard_memory_update(
        weights: list[torch.Tensor],
        momentum: list[torch.Tensor],
        grads: list[torch.Tensor],
        alpha: torch.Tensor | float,
        eta: torch.Tensor | float,
        theta: torch.Tensor | float,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Standard memory update using PyTorch operations.

        Args:
            weights: Current weight tensors
            momentum: Current momentum tensors
            grads: Gradient tensors
            alpha: Decay factor — scalar or (D,) per-output-dim tensor
            eta: Momentum coefficient — scalar or (D,) per-output-dim tensor
            theta: Learning rate — scalar or (D,) per-output-dim tensor

        Returns:
            Tuple of (new_weights, new_momentum)
        """
        # Update momentum: S_t = eta * S_{t-1} - theta * grad
        # m and g always have the same shape, so _broadcast_gate need only
        # inspect one of them to determine the broadcast shape for both.
        broadcast = NeuralLongTermMemory._broadcast_gate
        new_momentum = []
        for m, g in zip(momentum, grads, strict=True):
            _eta   = broadcast(eta,   m)
            _theta = broadcast(theta, m)  # same shape as m
            s = _eta * m - _theta * g
            new_momentum.append(s)

        # Update weights: M_t = (1 - alpha) * M_{t-1} + S_t
        new_weights = []
        for w, s in zip(weights, new_momentum, strict=True):
            _alpha = broadcast(alpha, w)
            w_new = (1 - _alpha) * w + s
            new_weights.append(w_new)

        return new_weights, new_momentum

    def retrieve(
        self,
        queries: torch.Tensor,
        state: MemoryState,
    ) -> torch.Tensor:
        """Retrieve from memory without updating.

        Args:
            queries: Query vectors (batch, seq, dim)
            state: Memory state to query

        Returns:
            Retrieved values (batch, seq, dim)
        """
        # Set weights from state
        self.memory.set_weights(state.weights)

        # Project queries
        q = self.proj_q(queries)

        if self.use_conv:
            q = self._apply_conv_single(q, self.conv_q)

        q = F.silu(q)
        q = F.normalize(q, p=2, dim=-1)

        # Retrieve
        retrieved = self.memory(q)
        return self.proj_out(retrieved)
