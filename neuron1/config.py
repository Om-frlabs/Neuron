"""NEURON-1 Model Configuration — AUDIT-HARDENED."""
from dataclasses import dataclass


@dataclass
class Neuron1Config:
    """Configuration for the NEURON-1 Dendritic Predictive Network.

    AUDIT CHANGES:
      - d_bottleneck: 64 → 96 (ablation showed 64 was harmful; 96 = 2.67× compression)
      - use_workspace: configurable (for ablation)
    """
    # Tokenizer / Embedding
    vocab_size: int = 4096
    d_model: int = 256

    # Layer counts
    n_fast_layers: int = 4
    n_slow_layers: int = 4

    # Dendritic Mixer
    n_dendrites: int = 4

    # Delta Memory (fast-weight state)
    d_state: int = 64

    # Global Workspace Bottleneck
    # AUDIT FIX: widened from 64 to 96 — 64 was too aggressive (ablation showed harm)
    d_bottleneck: int = 96
    use_workspace: bool = True

    # Hybrid Architecture (Frontier-style Attention Injection)
    use_hybrid_attention: bool = True
    n_attention_heads: int = 4

    # Mixture of Experts (MoE)
    use_moe: bool = True
    n_experts: int = 8
    n_active_experts: int = 2
    moe_loss_weight: float = 0.01

    # FFN
    ffn_ratio: float = 2.0

    # Temporal hierarchy
    fast_strides: tuple = (1, 1, 2, 2)
    total_stride: int = 4

    # CompNorm
    compnorm_init_temp: float = 5.0

    # RoPE (now used inside DeltaMemory only, at d_state dims)
    rope_base: float = 10000.0

    # Training
    max_seq_len: int = 512
    tie_weights: bool = True

    def __post_init__(self):
        assert len(self.fast_strides) == self.n_fast_layers, (
            f"fast_strides length ({len(self.fast_strides)}) must match "
            f"n_fast_layers ({self.n_fast_layers})"
        )
        assert self.d_model % self.n_dendrites == 0, (
            f"d_model ({self.d_model}) must be divisible by "
            f"n_dendrites ({self.n_dendrites})"
        )
