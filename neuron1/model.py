"""NEURON-1: Full Dendritic Predictive Network — AUDIT-HARDENED.

AUDIT FIXES:
  1. Removed RoPE from embedding — RoPE is now applied inside DeltaMemory
     to Q/K projections only (where relative position matters for
     associative memory reads). Slow layers use GatedLRU decay-position
     coupling for implicit positional encoding.

  2. Global Workspace widened to d_bottleneck=96 and made optional via
     config.use_workspace flag.

  3. Added gradual freeze with val-loss gating.

  4. Fixed TemporalUpsample receiving skip from correct layer index.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuron1.config import Neuron1Config
from neuron1.layers import (
    CompNorm,
    FastLayer,
    GlobalWorkspace,
    SlowLayer,
    TemporalUpsample,
    AttentionLayer,
)


class Neuron1(nn.Module):
    """NEURON-1: Dendritic Predictive Network.

    A ~4.7M-parameter architecture designed for maximum per-parameter
    intelligence. Key innovations:
      1. Dendritic Sparse Mixing (K=4 dendrites, O(nd), per-token routing)
      2. Dual-Timescale Memory (fast delta-rule + slow RG-LRU)
      3. Predictive Residual Connections (unified bottom-up direction)
      4. Global Workspace Bottleneck (256→96→256, optional)
      5. TemporalUpsample with projected skip fusion
    """

    def __init__(self, config: Neuron1Config | None = None):
        super().__init__()
        if config is None:
            config = Neuron1Config()
        self.config = config

        # ── Embedding (no RoPE — position handled inside DeltaMemory) ──
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # ── Fast Layers (plastic, inference-time adaptation) ──
        self.fast_layers = nn.ModuleList([
            FastLayer(
                d_model=config.d_model,
                d_state=config.d_state,
                n_dendrites=config.n_dendrites,
                stride=config.fast_strides[i],
                ffn_ratio=config.ffn_ratio,
                compnorm_temp=config.compnorm_init_temp,
                use_moe=config.use_moe,
                n_experts=config.n_experts,
                n_active_experts=config.n_active_experts,
            )
            for i in range(config.n_fast_layers)
        ])

        # ── Temporal Upsample (stride recovery) ──
        self.upsample = TemporalUpsample(config.d_model, config.total_stride)

        self.use_workspace = config.use_workspace
        if self.use_workspace:
            self.workspace = GlobalWorkspace(
                config.d_model, config.d_bottleneck, n_heads=4
            )

        # ── Hybrid Attention Injection ──
        self.use_hybrid_attention = config.use_hybrid_attention
        if self.use_hybrid_attention:
            self.attention_layer = AttentionLayer(
                config.d_model, config.n_attention_heads, config.compnorm_init_temp
            )

        # ── Slow Layers (frozen backbone, stable dynamics) ──
        self.slow_layers = nn.ModuleList([
            SlowLayer(
                d_model=config.d_model,
                d_state=config.d_state,
                ffn_ratio=config.ffn_ratio,
                compnorm_temp=config.compnorm_init_temp,
                use_moe=config.use_moe,
                n_experts=config.n_experts,
                n_active_experts=config.n_active_experts,
            )
            for _ in range(config.n_slow_layers)
        ])

        # ── Output ──
        self.final_norm = CompNorm(config.d_model, config.compnorm_init_temp)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        skip_params = {"tau", "pred_gate", "alpha", "recurrence_weight",
                       "scale", "pos_decay_bias"}
        for name, param in self.named_parameters():
            if any(s in name for s in skip_params):
                continue
            if "weight" in name and param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(param)
        # Smaller init for delta memory projections
        for layer in self.fast_layers:
            if hasattr(layer, 'delta_mem'):
                nn.init.normal_(layer.delta_mem.proj_k.weight, std=0.01)
                nn.init.normal_(layer.delta_mem.proj_q.weight, std=0.01)
                nn.init.normal_(layer.delta_mem.proj_v.weight, std=0.01)

    def _find_last_stride1_index(self) -> int:
        idx = 0
        for i, layer in enumerate(self.fast_layers):
            if layer.stride == 1:
                idx = i
        return idx

    def forward(
        self,
        input_ids: torch.Tensor,
        fast_states: list[torch.Tensor | None] | None = None,
        slow_states: list[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None], list[torch.Tensor | None]]:
        """
        Args:
            input_ids: (B, T) token indices
            fast_states: list of DeltaMemory states, one per fast layer
            slow_states: list of GatedLRU states, one per slow layer
        Returns:
            logits, new_fast_states, new_slow_states
        """
        if fast_states is None:
            fast_states = [None] * self.config.n_fast_layers
        if slow_states is None:
            slow_states = [None] * self.config.n_slow_layers

        # ── Embedding (NO RoPE — position is handled inside DeltaMemory) ──
        x = self.embedding(input_ids)
        orig_len = x.shape[1]

        # ── Fast Layers ──
        prev_output = None
        new_fast_states = []
        skip_connection = None
        skip_idx = self._find_last_stride1_index()
        self.moe_loss = 0.0
        last_layer_memory_trace = None  # Use final deepest memory trace for MLA component

        for i, layer in enumerate(self.fast_layers):
            x, state, aux_loss, h_mem = layer(x, prev_output, fast_states[i])
            self.moe_loss += aux_loss
            last_layer_memory_trace = h_mem
            
            prev_output = x
            new_fast_states.append(state)

            if i == skip_idx:
                skip_connection = x

        # ── Global Workspace Latent Compression (MLA) ──
        # Operates purely in the heavily compressed strided state (T/4) to save immense KV FLOPs.
        if self.use_workspace:
            x = x + self.workspace(x, memory_trace=last_layer_memory_trace)

        # ── Hybrid Attention Injection (O((T/4)^2)) ──
        # Moved before upsampling to preserve the sub-quadratic efficiency of the architecture
        if self.use_hybrid_attention:
            x, _ = self.attention_layer(x, None)

        # ── Temporal Upsample (T/4 → T) ──
        x = self.upsample(x, skip_connection, orig_len)

        # ── Slow Layers ──
        new_slow_states = []
        for i, layer in enumerate(self.slow_layers):
            x, state, aux_loss = layer(x, slow_states[i])
            self.moe_loss += aux_loss
            new_slow_states.append(state)

        # ── Output ──
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, new_fast_states, new_slow_states

    def freeze_slow_layers(self):
        for layer in self.slow_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_slow_layers(self):
        for layer in self.slow_layers:
            for param in layer.parameters():
                param.requires_grad = True

    def freeze_slow_layers_gradual(self, fraction: float = 1.0):
        """Gradual freeze: LRU first, then FFN.

        AUDIT FIX: allows staged freezing instead of binary on/off.
          fraction < 0.5: everything trainable
          0.5 <= fraction < 1.0: LRU frozen, FFN+norms trainable
          fraction >= 1.0: all frozen
        """
        for layer in self.slow_layers:
            for name, param in layer.named_parameters():
                if 'lru' in name:
                    param.requires_grad = fraction < 0.5
                else:
                    param.requires_grad = fraction < 1.0

    def count_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self) -> dict[str, int]:
        breakdown = {}
        for name, module in self.named_children():
            count = sum(p.numel() for p in module.parameters())
            breakdown[name] = count
        return breakdown