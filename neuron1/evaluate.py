"""NEURON-1 Evaluation Harness.

Lightweight probing metrics for Week 2 ablation validation:
  1. Validation loss (primary, computed in ablation runner)
  2. Activation sparsity (CompNorm effectiveness)
  3. Memory retrieval accuracy (DeltaMemory probe)
  4. Compositional generalization (novel combination test)
  5. Workspace compression ratio

These are designed to run fast (<10s) as diagnostic probes,
not as full TURING-NANO benchmarks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuron1.model import Neuron1
from neuron1.data import SimpleTokenizer


@torch.no_grad()
def measure_activation_sparsity(model: Neuron1, sample_ids: torch.Tensor) -> dict:
    """Measure activation sparsity at each layer.

    CompNorm should produce increasingly sparse activations at lower
    temperatures. Returns per-layer sparsity (fraction of activations < 0.01).
    """
    model.eval()
    sparsities = {}
    hooks = []
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output
            activations[name] = x.detach()
        return hook

    # Hook all layers
    for i, layer in enumerate(model.fast_layers):
        hooks.append(layer.register_forward_hook(make_hook(f"fast_{i}")))
    for i, layer in enumerate(model.slow_layers):
        hooks.append(layer.register_forward_hook(make_hook(f"slow_{i}")))

    model(sample_ids)

    for name, act in activations.items():
        near_zero = (act.abs() < 0.01).float().mean().item()
        sparsities[name] = near_zero

    for h in hooks:
        h.remove()

    return sparsities


@torch.no_grad()
def measure_memory_retrieval(model: Neuron1, tokenizer: SimpleTokenizer) -> dict:
    """Probe DeltaMemory retrieval: can the model reconstruct patterns?

    Creates a simple pattern (A B C ... A B ?) and checks if the model
    assigns higher probability to C at the query position.

    This tests in-context associative binding, the core DeltaMemory function.
    """
    model.eval()

    # Create patterns: "X Y Z ... X Y" → should predict Z
    n_trials = 10
    correct = 0

    for trial in range(n_trials):
        # Create tokens: A B C <padding> A B → should predict C
        a, b, c = trial + 10, trial + 50, trial + 100  # arbitrary tokens
        context = [1] + [a, b, c] * 3 + [a, b]  # BOS + pattern×3 + query
        ids = torch.tensor([context], dtype=torch.long)

        logits, _, _ = model(ids)
        predicted = logits[0, -1].argmax().item()
        if predicted == c:
            correct += 1

    return {
        "retrieval_accuracy": correct / n_trials,
        "correct": correct,
        "total": n_trials,
    }


@torch.no_grad()
def measure_workspace_compression(model: Neuron1, sample_ids: torch.Tensor) -> dict:
    """Measure the information compression in the Global Workspace.

    Computes:
    - Mean activation magnitude in the 64-dim bottleneck
    - Effective rank (via singular values) of the bottleneck
    - Entropy of the bottleneck distribution
    """
    model.eval()
    workspace_output = []

    def hook(module, input, output):
        workspace_output.append(output.detach())

    h = model.workspace.register_forward_hook(hook)
    model(sample_ids)
    h.remove()

    if not workspace_output:
        return {"compression_ratio": 0, "effective_rank": 0}

    z = workspace_output[0]  # (B, T, d_bottleneck)
    B, T, D = z.shape

    # Mean activation
    mean_activation = z.abs().mean().item()

    # Effective rank via singular values
    z_flat = z.reshape(-1, D)  # (B*T, D)
    if z_flat.shape[0] > 1:
        _, S, _ = torch.linalg.svd(z_flat[:min(z_flat.shape[0], 100)], full_matrices=False)
        S_norm = S / S.sum()
        entropy = -(S_norm * S_norm.clamp(min=1e-10).log()).sum().item()
        eff_rank = torch.exp(torch.tensor(entropy)).item()
    else:
        eff_rank = D

    return {
        "mean_activation": mean_activation,
        "effective_rank": eff_rank,
        "bottleneck_dim": D,
        "compression_ratio": model.config.d_model / D,
    }


@torch.no_grad()
def run_all_probes(
    model: Neuron1,
    tokenizer: SimpleTokenizer,
    seq_len: int = 128,
) -> dict:
    """Run all evaluation probes and return combined metrics."""
    # Create sample input
    sample_text = "Once upon a time there was a little cat who liked to play in the garden."
    sample_tokens = tokenizer.encode(sample_text)
    # Pad or truncate to seq_len
    if len(sample_tokens) < seq_len:
        sample_tokens = sample_tokens + [0] * (seq_len - len(sample_tokens))
    else:
        sample_tokens = sample_tokens[:seq_len]

    sample_ids = torch.tensor([sample_tokens], dtype=torch.long)

    results = {}
    results["sparsity"] = measure_activation_sparsity(model, sample_ids)
    results["memory"] = measure_memory_retrieval(model, tokenizer)
    results["workspace"] = measure_workspace_compression(model, sample_ids)

    return results


def print_probe_report(results: dict):
    """Pretty-print probe results."""
    print(f"\n{'='*60}")
    print("  EVALUATION PROBES")
    print(f"{'='*60}")

    print("\n  Activation Sparsity (fraction near zero):")
    for name, val in results["sparsity"].items():
        bar = "#" * int(val * 40)
        print(f"    {name:<12s}: {val:.3f} |{bar}")

    print(f"\n  Memory Retrieval:")
    m = results["memory"]
    print(f"    Accuracy: {m['correct']}/{m['total']} = {m['retrieval_accuracy']:.1%}")

    print(f"\n  Workspace Compression:")
    w = results["workspace"]
    print(f"    Compression ratio: {w['compression_ratio']:.1f}x ({w['bottleneck_dim']}d)")
    print(f"    Effective rank: {w['effective_rank']:.1f} / {w['bottleneck_dim']}")
    print(f"    Mean activation: {w['mean_activation']:.4f}")
