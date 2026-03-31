"""FLOP Profiler for NEURON-1 components.

Computes analytical FLOP counts per token for each component
to validate against the architecture spec target of ~3.5M FLOPs/token.
"""
import torch
from neuron1.config import Neuron1Config


def flop_breakdown(config: Neuron1Config | None = None) -> dict[str, dict]:
    """Compute analytical FLOPs per token for each component.

    Returns dict mapping component name → {flops, detail}.
    All counts assume a single token (T=1) at inference.
    """
    if config is None:
        config = Neuron1Config()

    d = config.d_model
    d_s = config.d_state
    d_b = config.d_bottleneck
    K = config.n_dendrites
    d_branch = d // K
    V = config.vocab_size
    ffn_h = int(d * config.ffn_ratio)

    results = {}

    # ── Embedding lookup: 0 FLOPs (just indexing) ──
    results["embedding"] = {
        "flops": 0,
        "detail": "Table lookup, no multiply-adds",
    }

    # ── RoPE: 6d multiplies per token ──
    results["rope"] = {
        "flops": 6 * d,
        "detail": f"2 splits × 3 ops (mul, mul, add) × d/2 = 6×{d}",
    }

    # ── Per Fast Layer ──
    # CompNorm: softmax(x/τ) + element-wise multiply
    compnorm_flops = 5 * d  # exp + sum + div + 2 mul
    # DendriticMixer: K × (gate: d→d_branch matmul + sigmoid + branch mul + transform: d_branch→d matmul + SiLU)
    dendrite_per_k = (d * d_branch) + d_branch + (d_branch * d) + (d_branch * d) + d
    dendrite_total = K * dendrite_per_k + d  # + weighted sum
    # DeltaMemory: proj_k + proj_v + proj_q + bmm read + surprise + delta write + proj_out
    delta_flops = 3 * (d * d_s) + (d_s * d_s) + d + (d_s * d_s) + (d_s * d)
    # FFN: d→ffn_h + SiLU + ffn_h→d
    ffn_flops = (d * ffn_h) + ffn_h + (ffn_h * d)
    # Predictor: d→d matmul
    predictor_flops = d * d

    fast_layer_flops = (
        2 * compnorm_flops +  # norm1 + norm2
        dendrite_total +
        delta_flops +
        ffn_flops +
        predictor_flops
    )

    results["fast_layer_single"] = {
        "flops": fast_layer_flops,
        "detail": (
            f"CompNorm×2={2*compnorm_flops}, DendriticMixer={dendrite_total}, "
            f"DeltaMemory={delta_flops}, FFN={ffn_flops}, Predictor={predictor_flops}"
        ),
    }
    results["fast_layers_total"] = {
        "flops": fast_layer_flops * config.n_fast_layers,
        "detail": f"{fast_layer_flops} × {config.n_fast_layers} layers",
    }

    # ── TemporalUpsample ──
    upsample_flops = (d * d) + (2 * d * d) + d  # interpolate + skip_gate + sigmoid
    results["upsample"] = {
        "flops": upsample_flops,
        "detail": f"interpolate d×d={d*d}, skip_gate 2d×d={2*d*d}",
    }

    # ── GlobalWorkspace ──
    gw_flops = (d * d_b) + compnorm_flops + d_b + (d_b * d)
    results["workspace"] = {
        "flops": gw_flops,
        "detail": f"compress={d*d_b}, CompNorm={compnorm_flops}, SiLU={d_b}, expand={d_b*d}",
    }

    # ── Per Slow Layer ──
    # GatedLRU: input_proj d→2*d_s + sigmoid + tanh + recurrence + output_proj d_s→d
    lru_flops = (d * 2 * d_s) + 2 * d_s + d_s + (d_s * d)
    slow_layer_flops = 2 * compnorm_flops + lru_flops + ffn_flops

    results["slow_layer_single"] = {
        "flops": slow_layer_flops,
        "detail": (
            f"CompNorm×2={2*compnorm_flops}, GatedLRU={lru_flops}, FFN={ffn_flops}"
        ),
    }
    results["slow_layers_total"] = {
        "flops": slow_layer_flops * config.n_slow_layers,
        "detail": f"{slow_layer_flops} × {config.n_slow_layers} layers",
    }

    # ── Final CompNorm + LM Head ──
    results["final_norm"] = {"flops": compnorm_flops, "detail": f"CompNorm d={d}"}
    results["lm_head"] = {"flops": d * V, "detail": f"{d}×{V} matmul"}

    # ── Total ──
    total = (
        results["embedding"]["flops"] +
        results["rope"]["flops"] +
        results["fast_layers_total"]["flops"] +
        results["upsample"]["flops"] +
        results["workspace"]["flops"] +
        results["slow_layers_total"]["flops"] +
        results["final_norm"]["flops"] +
        results["lm_head"]["flops"]
    )
    results["TOTAL"] = {"flops": total, "detail": "Sum of all components"}

    return results


def print_flop_report(config: Neuron1Config | None = None):
    """Print a formatted FLOP breakdown report."""
    breakdown = flop_breakdown(config)
    print("=" * 70)
    print("NEURON-1 FLOP BREAKDOWN (per token)")
    print("=" * 70)

    for name, info in breakdown.items():
        flops = info["flops"]
        if flops >= 1_000_000:
            fstr = f"{flops/1_000_000:.2f}M"
        elif flops >= 1_000:
            fstr = f"{flops/1_000:.1f}K"
        else:
            fstr = str(flops)
        print(f"  {name:25s}: {fstr:>10s}   ({info['detail']})")

    total = breakdown["TOTAL"]["flops"]
    print(f"\n  Target: ~3.5M FLOPs/token")
    print(f"  Actual: {total/1_000_000:.2f}M FLOPs/token")
    print(f"  Status: {'PASS' if total < 5_000_000 else 'OVER BUDGET'}")


if __name__ == "__main__":
    print_flop_report()
