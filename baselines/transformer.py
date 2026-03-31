"""Baseline Transformer (~5M params) for NEURON-1 comparison.

A standard pre-norm transformer with:
  - Multi-head causal self-attention
  - RMSNorm (modern standard)
  - SiLU FFN
  - RoPE positional encoding
  - Weight tying (embedding ↔ lm_head)

Matched to NEURON-1's parameter budget (~4.7M params).

Architecture:
  d_model=256, n_heads=4, n_layers=6, ffn_ratio=2.67, vocab=4096
  → ~4.7M params (within 5% of NEURON-1)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].unsqueeze(0),
            self.sin_cached[:seq_len].unsqueeze(0),
        )


def apply_rotary_emb(x, cos, sin):
    d_half = x.shape[-1] // 2
    x1, x2 = x[..., :d_half], x[..., d_half:]
    return torch.cat([
        x1 * cos[..., :d_half] - x2 * sin[..., :d_half],
        x2 * cos[..., :d_half] + x1 * sin[..., :d_half],
    ], dim=-1)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, n_heads, head_dim)

        # RoPE on Q and K (per-head)
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        cos_h = cos[:, :, :self.head_dim]
        sin_h = sin[:, :, :self.head_dim]
        q = apply_rotary_emb(q, cos_h.unsqueeze(1), sin_h.unsqueeze(1))
        k = apply_rotary_emb(k, cos_h.unsqueeze(1), sin_h.unsqueeze(1))

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, d_model: int, n_heads: int, ffn_ratio: float = 2.67):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)

        ffn_hidden = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.SiLU(),
            nn.Linear(ffn_hidden, d_model),
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class BaselineTransformer(nn.Module):
    """Standard transformer matched to NEURON-1's parameter budget.

    ~4.7M params with d_model=256, n_heads=4, n_layers=6, ffn_ratio=2.67.
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        ffn_ratio: float = 2.67,
        max_seq_len: int = 512,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len=max_seq_len)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_ratio)
            for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) token indices
        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.embedding(input_ids)
        T = x.shape[1]
        cos, sin = self.rope(T)

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.final_norm(x)
        return self.lm_head(x)

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
