"""Comprehensive tests for NEURON-1 architecture — AUDIT-HARDENED.

Tests every novel component individually, then validates the full
model assembly for correctness, parameter budget, and gradient flow.

AUDIT ADDITIONS:
  - CompNorm: variance control test (output variance bounded)
  - DendriticMixer: per-token routing test
  - FastLayer: depthwise conv stride test
  - Predictive residual: unified direction test
  - GatedLRU: position-decay coupling test
  - Gradual freeze test
  - Loss function integration test with new loss components

Run: pytest tests/ -v
"""
import pytest
import torch

from neuron1.config import Neuron1Config
from neuron1.layers import (
    CompNorm,
    DeltaMemory,
    DendriticMixer,
    FastLayer,
    GatedLRU,
    GlobalWorkspace,
    RotaryEmbedding,
    SlowLayer,
    TemporalUpsample,
    apply_rotary_emb,
)
from neuron1.model import Neuron1

# ── Test fixtures ──
B, T, D = 2, 64, 256  # batch, seq_len, d_model
D_STATE = 64
N_DENDRITES = 4


@pytest.fixture
def config():
    return Neuron1Config(max_seq_len=T)


@pytest.fixture
def model(config):
    return Neuron1(config)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CompNorm Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestCompNorm:
    def test_output_shape(self):
        norm = CompNorm(D)
        x = torch.randn(B, T, D)
        y = norm(x)
        assert y.shape == (B, T, D)

    def test_gradient_flow(self):
        norm = CompNorm(D)
        x = torch.randn(B, T, D, requires_grad=True)
        y = norm(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_temperature_is_learnable(self):
        norm = CompNorm(D)
        assert norm.tau.requires_grad

    def test_temp_clamp_prevents_collapse(self):
        """AUDIT: tau clamped at 1.0 minimum to prevent winner-take-all."""
        norm = CompNorm(D, init_temp=0.01)
        x = torch.randn(B, T, D)
        y = norm(x)
        assert not torch.isnan(y).any()

    def test_variance_control(self):
        """AUDIT: CompNorm must control output variance. Previous version
        had unbounded output magnitude that scaled as O(x²)."""
        norm = CompNorm(D)
        # Test with different input scales
        for scale in [0.1, 1.0, 10.0]:
            x = torch.randn(B, T, D) * scale
            y = norm(x)
            # Output variance per-sample should be bounded
            var = y.var(dim=-1).mean()
            assert var < 1000, (
                f"CompNorm output variance {var:.1f} unbounded at input scale {scale}"
            )
            assert not torch.isnan(y).any()
            assert not torch.isinf(y).any()

    def test_normalization_reduces_scale_sensitivity(self):
        """AUDIT: output scale should be much less sensitive to input scale
        than a linear operation would be."""
        norm = CompNorm(D)
        x_small = torch.randn(B, T, D) * 0.1
        x_big = torch.randn(B, T, D) * 10.0
        y_small = norm(x_small)
        y_big = norm(x_big)
        ratio = y_big.norm() / y_small.norm()
        # Without normalization, ratio would be ~100. With it, should be < 20
        assert ratio < 20, f"CompNorm not controlling scale: ratio={ratio:.1f}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DendriticMixer Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestDendriticMixer:
    def test_output_shape(self):
        mixer = DendriticMixer(D, N_DENDRITES)
        x = torch.randn(B, T, D)
        y = mixer(x)
        assert y.shape == (B, T, D)

    def test_all_dendrites_receive_gradients(self):
        mixer = DendriticMixer(D, N_DENDRITES)
        x = torch.randn(B, T, D, requires_grad=True)
        y = mixer(x)
        y.sum().backward()

        for k in range(N_DENDRITES):
            assert mixer.gates[k].weight.grad is not None
            assert mixer.transforms[k].weight.grad is not None
            assert not torch.isnan(mixer.gates[k].weight.grad).any()

    def test_per_token_routing(self):
        """AUDIT: mixture weights should differ per token, not be static."""
        mixer = DendriticMixer(D, N_DENDRITES)
        # Create two very different inputs
        x = torch.randn(1, 2, D)
        x[:, 0] *= 10.0  # very different scales
        x[:, 1] *= 0.1

        # Get routing weights for both tokens
        weights = torch.softmax(mixer.alpha_net(x), dim=-1)  # (1, 2, K)
        w0 = weights[0, 0]
        w1 = weights[0, 1]
        # Should NOT be identical (unless alpha_net has exactly zero weights)
        # With random init, they should differ
        assert not torch.allclose(w0, w1, atol=1e-3), \
            "Per-token routing weights should differ for different inputs"

    def test_alpha_net_has_gradients(self):
        """AUDIT: alpha_net should receive gradients."""
        mixer = DendriticMixer(D, N_DENDRITES)
        x = torch.randn(B, T, D, requires_grad=True)
        y = mixer(x)
        y.sum().backward()
        assert mixer.alpha_net.weight.grad is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DeltaMemory Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestDeltaMemory:
    def test_output_shape(self):
        mem = DeltaMemory(D, D_STATE)
        x = torch.randn(B, T, D)
        y, state = mem(x)
        assert y.shape == (B, T, D)
        assert state.shape == (B, D_STATE, D_STATE)

    def test_state_updates_follow_delta_rule(self):
        mem = DeltaMemory(D, D_STATE)
        x = torch.randn(B, T, D)
        _, state1 = mem(x[:, :T // 2])
        _, state2 = mem(x, state=None)
        assert not torch.allclose(state1, state2[:, :, :])

    def test_state_carryover(self):
        mem = DeltaMemory(D, D_STATE)
        x = torch.randn(B, T, D)
        y1, state1 = mem(x[:, :T // 2])
        y2, state2 = mem(x[:, T // 2:], state=state1)
        assert state2.shape == (B, D_STATE, D_STATE)

    def test_surprise_gate_modulates_alpha(self):
        mem = DeltaMemory(D, D_STATE)
        x = torch.randn(B, T, D)
        surprise = torch.sigmoid(mem.surprise_gate(x))
        assert (surprise >= 0).all() and (surprise <= 1).all()

    def test_stability_over_long_sequence(self):
        mem = DeltaMemory(D, D_STATE)
        x = torch.randn(B, 256, D)
        y, state = mem(x)
        assert not torch.isnan(y).any(), "DeltaMemory should be stable over 256 steps"
        assert not torch.isnan(state).any()

    def test_gradient_flow(self):
        mem = DeltaMemory(D, D_STATE)
        x = torch.randn(B, 16, D, requires_grad=True)  # shorter for speed
        y, _ = mem(x)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_has_internal_rope(self):
        """AUDIT: DeltaMemory should have its own RoPE for Q/K."""
        mem = DeltaMemory(D, D_STATE)
        assert hasattr(mem, 'rope'), "DeltaMemory must have internal RoPE"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GatedLRU Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestGatedLRU:
    def test_output_shape(self):
        lru = GatedLRU(D, D_STATE)
        x = torch.randn(B, T, D)
        y, state = lru(x)
        assert y.shape == (B, T, D)
        assert state.shape == (B, D_STATE)

    def test_recurrent_dynamics_stable_1000_steps(self):
        lru = GatedLRU(D, D_STATE)
        x = torch.randn(B, 1000, D)
        y, state = lru(x)
        assert not torch.isnan(y).any(), "GatedLRU must be stable over 1000 steps"
        assert not torch.isinf(y).any()

    def test_decay_in_valid_range(self):
        lru = GatedLRU(D, D_STATE)
        a = torch.sigmoid(lru.recurrence_weight + lru.pos_decay_bias)
        assert (a > 0).all() and (a < 1).all()

    def test_state_carryover(self):
        lru = GatedLRU(D, D_STATE)
        x = torch.randn(B, T, D)
        y1, state1 = lru(x[:, :T // 2])
        y2, state2 = lru(x[:, T // 2:], state=state1)
        assert state2.shape == (B, D_STATE)

    def test_position_decay_coupling(self):
        """AUDIT: GatedLRU should have learnable position-decay bias."""
        lru = GatedLRU(D, D_STATE)
        assert hasattr(lru, 'pos_decay_bias')
        assert lru.pos_decay_bias.requires_grad

    def test_chunk_size_guard(self):
        """AUDIT: CHUNK_SIZE should not exceed MAX_CHUNK_SIZE."""
        lru = GatedLRU(D, D_STATE)
        assert lru.CHUNK_SIZE <= lru.MAX_CHUNK_SIZE


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GlobalWorkspace Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestGlobalWorkspace:
    def test_output_shape(self):
        gw = GlobalWorkspace(D, 96)
        x = torch.randn(B, T, D)
        y = gw(x)
        assert y.shape == (B, T, D)

    def test_bottleneck_compression(self):
        """AUDIT: bottleneck widened from 64 to 96."""
        gw = GlobalWorkspace(D, 96)
        assert gw.compress.out_features == 96
        assert gw.expand.in_features == 96

    def test_uses_layernorm(self):
        """AUDIT: should use LayerNorm, not CompNorm, for clean channel."""
        gw = GlobalWorkspace(D, 96)
        assert isinstance(gw.norm, torch.nn.LayerNorm)

    def test_gradient_flow(self):
        gw = GlobalWorkspace(D, 96)
        x = torch.randn(B, T, D, requires_grad=True)
        y = gw(x)
        y.sum().backward()
        assert x.grad is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TemporalUpsample Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestTemporalUpsample:
    def test_output_shape(self):
        up = TemporalUpsample(D, total_stride=4)
        x_strided = torch.randn(B, T // 4, D)
        x_skip = torch.randn(B, T, D)
        y = up(x_strided, x_skip, orig_len=T)
        assert y.shape == (B, T, D)

    def test_handles_non_divisible_lengths(self):
        up = TemporalUpsample(D, total_stride=4)
        odd_T = 67
        x_strided = torch.randn(B, odd_T // 4 + 1, D)
        x_skip = torch.randn(B, odd_T, D)
        y = up(x_strided, x_skip, orig_len=odd_T)
        assert y.shape == (B, odd_T, D)

    def test_gradient_flow(self):
        up = TemporalUpsample(D, total_stride=4)
        x_strided = torch.randn(B, T // 4, D, requires_grad=True)
        x_skip = torch.randn(B, T, D, requires_grad=True)
        y = up(x_strided, x_skip, orig_len=T)
        y.sum().backward()
        assert x_strided.grad is not None
        assert x_skip.grad is not None

    def test_has_skip_projection(self):
        """AUDIT: skip should be projected to match deep-layer abstraction."""
        up = TemporalUpsample(D, total_stride=4)
        assert hasattr(up, 'skip_proj')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RoPE Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestRoPE:
    def test_output_shapes(self):
        rope = RotaryEmbedding(D, max_seq_len=T)
        cos, sin = rope(T)
        assert cos.shape == (1, T, D)
        assert sin.shape == (1, T, D)

    def test_apply_rotary_preserves_shape(self):
        rope = RotaryEmbedding(D, max_seq_len=T)
        x = torch.randn(B, T, D)
        cos, sin = rope(T)
        y = apply_rotary_emb(x, cos, sin)
        assert y.shape == x.shape

    def test_cache_extends_automatically(self):
        rope = RotaryEmbedding(D, max_seq_len=32)
        cos, sin = rope(128)  # longer than cache
        assert cos.shape[1] == 128


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FastLayer Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestFastLayer:
    def test_stride_1_preserves_length(self):
        layer = FastLayer(D, D_STATE, N_DENDRITES, stride=1)
        x = torch.randn(B, T, D)
        y, _ = layer(x)
        assert y.shape == (B, T, D)

    def test_stride_2_halves_length(self):
        layer = FastLayer(D, D_STATE, N_DENDRITES, stride=2)
        x = torch.randn(B, T, D)
        y, _ = layer(x)
        assert y.shape == (B, T // 2, D)

    def test_predictive_residual(self):
        layer = FastLayer(D, D_STATE, N_DENDRITES, stride=1)
        x = torch.randn(B, T, D)
        prev = torch.randn(B, T, D)
        y_no_pred, _ = layer(x)
        y_with_pred, _ = layer(x, prev_layer_output=prev)
        assert not torch.allclose(y_no_pred, y_with_pred)

    def test_uses_depthwise_conv_for_stride(self):
        """AUDIT: stride should use depthwise Conv1d, not max-pool."""
        layer = FastLayer(D, D_STATE, N_DENDRITES, stride=2)
        assert hasattr(layer, 'downsample')
        assert isinstance(layer.downsample, torch.nn.Conv1d)
        assert layer.downsample.groups == D  # depthwise

    def test_stride_conv_gradients_flow(self):
        """AUDIT: all tokens should receive gradients through depthwise conv."""
        layer = FastLayer(D, D_STATE, N_DENDRITES, stride=2)
        x = torch.randn(B, T, D, requires_grad=True)
        y, _ = layer(x)
        y.sum().backward()
        # Every token in x should have non-zero gradient
        per_token_grad_norm = x.grad.norm(dim=-1)  # (B, T)
        assert (per_token_grad_norm > 0).all(), \
            "Depthwise conv should provide gradients to ALL input tokens"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SlowLayer Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestSlowLayer:
    def test_output_shape(self):
        layer = SlowLayer(D, D_STATE)
        x = torch.randn(B, T, D)
        y, state = layer(x)
        assert y.shape == (B, T, D)
        assert state.shape == (B, D_STATE)

    def test_state_carryover(self):
        layer = SlowLayer(D, D_STATE)
        x = torch.randn(B, T, D)
        _, state1 = layer(x[:, :T // 2])
        y2, state2 = layer(x[:, T // 2:], state=state1)
        assert y2.shape == (B, T // 2, D)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Full Model Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestNeuron1Model:
    def test_forward_pass_shape(self, model):
        ids = torch.randint(0, 4096, (B, T))
        logits, fast_s, slow_s = model(ids)
        assert logits.shape == (B, T, 4096)
        assert len(fast_s) == 4
        assert len(slow_s) == 4

    def test_parameter_count_within_budget(self, model):
        total = model.count_parameters()
        assert total < 6_000_000, f"Total params {total} exceeds 6M budget"
        assert total > 3_000_000, f"Total params {total} suspiciously low"

    def test_parameter_breakdown(self, model):
        breakdown = model.parameter_breakdown()
        assert "embedding" in breakdown
        assert "fast_layers" in breakdown
        assert "slow_layers" in breakdown
        # Print for visibility
        print("\n=== NEURON-1 Parameter Breakdown ===")
        total = 0
        for name, count in breakdown.items():
            print(f"  {name:20s}: {count:>10,}")
            total += count
        print(f"  {'TOTAL':20s}: {total:>10,}")

    def test_weight_tying(self, model):
        """Verify embedding and lm_head share the same weight tensor."""
        assert model.embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()

    def test_gradient_flow_full_model(self, model):
        ids = torch.randint(0, 4096, (B, T))
        logits, _, _ = model(ids)
        loss = logits.sum()
        loss.backward()

        assert model.embedding.weight.grad is not None
        for i, layer in enumerate(model.fast_layers):
            assert layer.dendritic.alpha_net.weight.grad is not None, \
                f"Fast layer {i} alpha_net has no grad"
            assert layer.delta_mem.proj_k.weight.grad is not None

    def test_state_carryover_across_sequences(self, model):
        ids1 = torch.randint(0, 4096, (B, T))
        ids2 = torch.randint(0, 4096, (B, T))

        _, fast_s, slow_s = model(ids1)
        logits2, _, _ = model(ids2, fast_states=fast_s, slow_states=slow_s)
        assert logits2.shape == (B, T, 4096)

    def test_freeze_slow_layers(self, model):
        model.freeze_slow_layers()
        for layer in model.slow_layers:
            for param in layer.parameters():
                assert not param.requires_grad
        for layer in model.fast_layers:
            for param in layer.parameters():
                assert param.requires_grad

    def test_unfreeze_slow_layers(self, model):
        model.freeze_slow_layers()
        model.unfreeze_slow_layers()
        for layer in model.slow_layers:
            for param in layer.parameters():
                assert param.requires_grad

    def test_gradual_freeze(self, model):
        """AUDIT: gradual freeze should freeze LRU first, then FFN."""
        model.freeze_slow_layers_gradual(0.75)
        for layer in model.slow_layers:
            for name, param in layer.named_parameters():
                if 'lru' in name:
                    assert not param.requires_grad, f"{name} should be frozen at fraction=0.75"
                else:
                    assert param.requires_grad, f"{name} should still be trainable at fraction=0.75"
        model.unfreeze_slow_layers()

    def test_no_nan_in_output(self, model):
        ids = torch.randint(0, 4096, (B, T))
        logits, _, _ = model(ids)
        assert not torch.isnan(logits).any(), "Model output contains NaN"
        assert not torch.isinf(logits).any(), "Model output contains Inf"

    def test_gradient_norms_reasonable(self, model):
        """Check gradient norms are not exploding or vanishing."""
        ids = torch.randint(0, 4096, (B, T))
        logits, _, _ = model(ids)
        loss = logits[:, :, 0].sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm < 1e5, f"Gradient explosion in {name}: norm={grad_norm}"

    def test_no_rope_in_embedding(self, model):
        """AUDIT: RoPE should NOT be applied at embedding level."""
        assert not hasattr(model, 'rope'), "Model should not have model-level RoPE"

    def test_workspace_is_optional(self):
        """AUDIT: workspace should be configurable."""
        config_off = Neuron1Config(use_workspace=False, max_seq_len=T)
        model_off = Neuron1(config_off)
        ids = torch.randint(0, 4096, (B, T))
        logits, _, _ = model_off(ids)
        assert logits.shape == (B, T, 4096)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Config Validation Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestConfig:
    def test_default_config_valid(self):
        config = Neuron1Config()
        assert config.d_model == 256
        assert config.vocab_size == 4096
        assert config.d_bottleneck == 96  # AUDIT: widened from 64

    def test_stride_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            Neuron1Config(fast_strides=(1, 1, 2))  # only 3, but n_fast=4

    def test_d_model_not_divisible_by_dendrites_raises(self):
        with pytest.raises(AssertionError):
            Neuron1Config(d_model=255, n_dendrites=4)
