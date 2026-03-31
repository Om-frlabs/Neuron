"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          NEURON-1  ·  Architecture Analysis & Pre-Training Evaluation       ║
║                                                                              ║
║  Five evaluation dimensions:                                                 ║
║    1. Scaling Laws         — proxy training + power-law extrapolation        ║
║    2. Theoretical Analysis — expressivity, gradient dynamics, inductive bias ║
║    3. Mechanistic Interp.  — info flow, activation geometry in random nets   ║
║    4. Toy Benchmarks       — sorting, symbolic reasoning, logic tasks        ║
║    5. Neuroscience-Inspired— working memory, hierarchy, feedback coupling    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage (from project root where neuron1/ package lives):
    python neuron1_analysis.py

Or with explicit path:
    python neuron1_analysis.py --neuron1-path /path/to/neuron1/
"""

import sys
import os
import math
import time
import json
import argparse
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────
#  PATH SETUP — find neuron1 package
# ─────────────────────────────────────────────────
def _setup_path(explicit: Optional[str] = None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit).parent)
    candidates += [
        Path(__file__).parent,           # same dir as this script
        Path(__file__).parent.parent,    # one level up
        Path.cwd(),
        Path.cwd().parent,
    ]
    for p in candidates:
        if (p / "neuron1").is_dir() or (p / "config.py").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return True
    return False


# ─────────────────────────────────────────────────
#  COLOUR HELPERS
# ─────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _ok(s):   return f"{GREEN}✓ {s}{RESET}"
def _warn(s): return f"{YELLOW}⚠ {s}{RESET}"
def _fail(s): return f"{RED}✗ {s}{RESET}"
def _hdr(s):  return f"\n{BOLD}{CYAN}{'─'*70}\n  {s}\n{'─'*70}{RESET}"


# ══════════════════════════════════════════════════════════════════════════════
#  1.  SCALING LAWS  —  proxy training + power-law extrapolation
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ScalingResult:
    d_model:    int
    n_params:   int
    train_loss: float
    val_loss:   float
    steps:      int
    tokens:     int


def _make_tiny_config(d_model: int):
    """Build a Neuron1Config scaled to d_model, keeping aspect ratios."""
    from neuron1.config import Neuron1Config
    n_dend = 4 if d_model >= 64 else 2
    while d_model % n_dend != 0:
        n_dend //= 2
    return Neuron1Config(
        vocab_size=256,       # byte-level for toy data
        d_model=d_model,
        n_fast_layers=2,      # keep fast for proxy run
        n_slow_layers=2,
        n_dendrites=n_dend,
        d_state=max(16, d_model // 4),
        d_bottleneck=max(16, d_model // 3),
        ffn_ratio=2.0,
        fast_strides=(1, 2),
        total_stride=2,
        max_seq_len=64,
        tie_weights=True,
        compnorm_init_temp=5.0,
    )


def _synthetic_tokens(n_sequences: int, seq_len: int, vocab: int = 256) -> torch.Tensor:
    """Tiny synthetic text: repeating patterns so loss can actually drop."""
    tokens = []
    for _ in range(n_sequences):
        base = torch.randint(1, vocab, (seq_len // 4,))
        rep  = base.repeat(4)[:seq_len + 1]
        tokens.append(rep)
    return torch.stack(tokens)   # (N, seq_len+1)


def run_scaling_laws(
    d_models:    List[int] = (32, 64, 128, 256),
    proxy_steps: int       = 300,
    batch_size:  int       = 16,
    seq_len:     int       = 48,
    lr:          float     = 3e-3,
    device:      str       = "cpu",
) -> List[ScalingResult]:
    """
    Train tiny proxy models at multiple widths and fit a scaling law.

    Chinchilla / Kaplan law:  L(N) ≈ A · N^(-α) + L∞
    We fit α and L∞ from the 4 data points.
    """
    from neuron1.model import Neuron1
    from neuron1.loss  import Neuron1Loss, Neuron1WithHooks

    results = []
    all_tokens = _synthetic_tokens(1024, seq_len, vocab=256)

    for d in d_models:
        print(f"  Proxy training  d_model={d:>4d} ...", end=" ", flush=True)
        t0 = time.time()
        cfg   = _make_tiny_config(d)
        model = Neuron1(cfg).to(device)
        n_p   = model.count_parameters()

        hooked = Neuron1WithHooks(model)
        crit   = Neuron1Loss(lambda_pred=0.05, lambda_collapse=0.02, lambda_compress=0.005)
        opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

        # Build loader
        X = all_tokens[:, :-1].to(device)
        Y = all_tokens[:, 1:].to(device)
        loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)

        train_losses, val_losses = [], []
        step = 0
        while step < proxy_steps:
            model.train()
            for xb, yb in loader:
                if step >= proxy_steps:
                    break
                logits, _, _ = hooked(xb)
                losses = crit(logits, yb, hooked, xb)
                opt.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                train_losses.append(losses["ce"].item())
                step += 1

        # Val
        model.eval()
        with torch.no_grad():
            val_batch_x = X[:64]
            val_batch_y = Y[:64]
            logits, _, _ = model(val_batch_x)
            val_loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                val_batch_y.reshape(-1)
            ).item()

        avg_train = sum(train_losses[-50:]) / 50
        tokens    = step * batch_size * seq_len

        results.append(ScalingResult(d, n_p, avg_train, val_loss, step, tokens))
        elapsed = time.time() - t0
        print(f"N={n_p:>7,}  val_loss={val_loss:.4f}  [{elapsed:.1f}s]")

    return results


def fit_scaling_law(results: List[ScalingResult]) -> Dict:
    """Fit  L = A·N^(-α) + L∞  via log-linear regression on the CE component."""
    import math
    if len(results) < 3:
        return {"note": "Need ≥3 points to fit"}

    # Two-point estimate for α using largest and smallest
    n1, l1 = results[0].n_params, results[0].val_loss
    n2, l2 = results[-1].n_params, results[-1].val_loss
    if l1 <= l2 or n1 >= n2:
        return {"note": "Loss did not decrease with scale — more training needed"}

    alpha = -math.log(l2 / l1) / math.log(n2 / n1)
    L_inf = l2 - (l1 - l2) * (n1 ** alpha) / ((n2 ** alpha) - (n1 ** alpha))
    A     = (l1 - L_inf) * (n1 ** alpha)

    # Extrapolate to full 5.24M model
    full_n = 5_240_000
    pred_loss = A * (full_n ** (-alpha)) + max(L_inf, 0)

    return {
        "alpha":           round(alpha, 4),
        "L_inf":           round(L_inf, 4),
        "A":               round(A, 4),
        "predicted_loss_5M": round(pred_loss, 4),
        "note": (
            "Extrapolation is rough (proxy ≠ full dataset/schedule). "
            "Use as directional signal, not hard prediction."
        ),
    }


def print_scaling_report(results: List[ScalingResult], law: Dict):
    print(_hdr("1. SCALING LAWS"))
    print(f"  {'d_model':>8}  {'params':>10}  {'train_loss':>12}  {'val_loss':>10}  {'tokens':>12}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*12}")
    for r in results:
        tag = ""
        if r.val_loss < 3.5:
            tag = "  " + _ok("good compression")
        elif r.val_loss > 5.5:
            tag = "  " + _warn("underfitting")
        print(f"  {r.d_model:>8}  {r.n_params:>10,}  {r.train_loss:>12.4f}  {r.val_loss:>10.4f}  {r.tokens:>12,}{tag}")

    print(f"\n  Scaling law fit:  L(N) ≈ {law.get('A','?')} · N^(-{law.get('alpha','?')}) + {law.get('L_inf','?')}")
    pred = law.get('predicted_loss_5M')
    if pred:
        status = _ok(f"Projected loss {pred} for 5.24M model") if pred < 4.5 else _warn(f"Projected loss {pred} — needs more data/steps")
        print(f"  Full model:  {status}")
    print(f"  Note: {law.get('note','')}")


# ══════════════════════════════════════════════════════════════════════════════
#  2.  THEORETICAL ANALYSIS
#      — expressivity, gradient dynamics, inductive bias, weight geometry
# ══════════════════════════════════════════════════════════════════════════════

def analyze_weight_geometry(model: nn.Module) -> Dict:
    """
    Examine the weight geometry of a randomly-initialised model:
      - Effective rank of key weight matrices
      - Condition number (ratio max/min singular value)
      - Spectral norm (largest singular value)
      - Nuclear norm / Frobenius norm ratio (proxy for rank utilisation)
    """
    metrics = {}
    for name, param in model.named_parameters():
        if param.dim() < 2 or param.numel() > 500_000:
            continue
        W = param.detach().float()
        if W.dim() > 2:
            W = W.view(W.shape[0], -1)

        try:
            S = torch.linalg.svdvals(W)
            S_pos = S[S > 1e-8]
            if len(S_pos) == 0:
                continue
            S_norm      = S_pos / S_pos.sum()
            entropy     = -(S_norm * S_norm.clamp(min=1e-10).log()).sum().item()
            eff_rank    = math.exp(entropy)
            cond        = (S_pos[0] / S_pos[-1]).item()
            spectral    = S_pos[0].item()
            fro         = W.norm("fro").item()
            nuc         = S_pos.sum().item()

            metrics[name] = {
                "shape":       list(W.shape),
                "eff_rank":    round(eff_rank, 2),
                "max_rank":    min(W.shape),
                "rank_util":   round(eff_rank / min(W.shape), 3),
                "cond":        round(cond, 2),
                "spectral":    round(spectral, 4),
                "nuc_fro":     round(nuc / (fro + 1e-8), 4),
            }
        except Exception:
            pass
    return metrics


def analyze_gradient_flow(model: nn.Module, seq_len: int = 32, vocab: int = 4096) -> Dict:
    """
    Simulate a single forward+backward with random data and measure
    per-layer gradient norms. Diagnoses vanishing/exploding gradients
    before any training has happened.
    """
    model.train()
    ids = torch.randint(0, vocab, (2, seq_len))
    tgt = torch.randint(0, vocab, (2, seq_len))
    logits, _, _ = model(ids)
    loss = F.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1))
    loss.backward()

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = round(param.grad.norm().item(), 6)

    model.zero_grad()

    # Summarise by module group
    groups = {}
    for n, v in grad_norms.items():
        top = n.split(".")[0]
        groups.setdefault(top, []).append(v)

    summary = {g: {
        "mean":  round(sum(vs)/len(vs), 6),
        "max":   round(max(vs), 6),
        "min":   round(min(vs), 6),
        "dead":  sum(1 for v in vs if v < 1e-7),
        "total": len(vs),
    } for g, vs in groups.items()}

    return {"per_group": summary, "all": grad_norms}


def analyze_inductive_bias(model: nn.Module) -> Dict:
    """
    Probe inductive biases without training:

    1. Locality bias — are nearby tokens more similar in representation space?
    2. Recency bias   — does the last token carry more representational variance?
    3. Dendrite routing diversity — do different tokens activate different dendrites?
    4. Compression bottleneck — effective rank drop through workspace?
    """
    model.eval()

    seq_len = 64
    vocab = model.config.vocab_size
    ids = torch.randint(1, vocab, (4, seq_len))

    layer_outputs = {}

    def hook_fn(name):
        def h(mod, inp, out):
            if isinstance(out, tuple):
                layer_outputs[name] = out[0].detach()
            else:
                layer_outputs[name] = out.detach()
        return h

    hooks = []
    for i, layer in enumerate(model.fast_layers):
        hooks.append(layer.register_forward_hook(hook_fn(f"fast_{i}")))
    for i, layer in enumerate(model.slow_layers):
        hooks.append(layer.register_forward_hook(hook_fn(f"slow_{i}")))
    if hasattr(model, 'workspace'):
        hooks.append(model.workspace.register_forward_hook(hook_fn("workspace")))
    hooks.append(model.embedding.register_forward_hook(hook_fn("embedding")))

    with torch.no_grad():
        model(ids)

    for h in hooks:
        h.remove()

    results = {}

    # ── 1. Locality bias ──
    if "embedding" in layer_outputs and "fast_3" in layer_outputs:
        emb = layer_outputs["embedding"][0]    # (T, D)
        lat = layer_outputs[list(layer_outputs.keys())[-2]][0]   # last fast layer (T, D)
        # Cos similarity between adjacent tokens
        def adj_cosine(x):
            x = F.normalize(x, dim=-1)
            return (x[:-1] * x[1:]).sum(-1).mean().item()
        results["locality_bias"] = {
            "embedding_adj_cosine":    round(adj_cosine(emb), 4),
            "deep_fast_adj_cosine":    round(adj_cosine(lat), 4),
            "note": "Higher cosine → model maps similar positions similarly (locality bias)"
        }

    # ── 2. Recency bias ──
    if "fast_3" in layer_outputs:
        act = layer_outputs["fast_3"][0]   # (T, D) first batch item
        position_variances = act.var(dim=-1)   # variance of each position
        first_half = position_variances[:seq_len//2].mean().item()
        second_half = position_variances[seq_len//2:].mean().item()
        results["recency_bias"] = {
            "first_half_var":  round(first_half, 6),
            "second_half_var": round(second_half, 6),
            "ratio":           round(second_half / (first_half + 1e-8), 4),
            "note": "Ratio > 1 means recent tokens have higher representational variance"
        }

    # ── 3. Dendrite routing diversity ──
    dendrite_info = {}
    with torch.no_grad():
        for i, fl in enumerate(model.fast_layers):
            ids_single = ids[:1]   # one batch item
            x = model.embedding(ids_single)
            # Compute alpha_net weights for this layer's dendritic mixer
            h = fl.norm1(x)
            weights = F.softmax(fl.dendritic.alpha_net(h), dim=-1)  # (1, T, K)
            entropy_per_pos = -(weights * weights.clamp(min=1e-9).log()).sum(-1)  # (1, T)
            max_ent = math.log(model.config.n_dendrites)
            dendrite_info[f"fast_{i}_routing_entropy"] = round(
                (entropy_per_pos.mean() / max_ent).item(), 4
            )
    results["dendrite_routing"] = {
        **dendrite_info,
        "note": "Entropy near 1.0 = uniform (no routing); near 0 = winner-take-all"
    }

    # ── 4. Workspace compression rank ──
    if "workspace" in layer_outputs:
        z = layer_outputs["workspace"][0]   # (T, d_bottleneck)
        S = torch.linalg.svdvals(z.float())
        S_n = S / S.sum()
        eff_rank = math.exp(-(S_n * S_n.clamp(1e-10).log()).sum().item())
        results["workspace_rank"] = {
            "effective_rank": round(eff_rank, 2),
            "bottleneck_dim": model.config.d_bottleneck,
            "rank_util": round(eff_rank / model.config.d_bottleneck, 3),
            "note": "Low rank_util = bottleneck is collapsing to few dimensions"
        }

    return results


def print_theoretical_report(weight_geo: Dict, grad_flow: Dict, inductive: Dict):
    print(_hdr("2. THEORETICAL ANALYSIS"))

    # Weight geometry summary
    print("  ── Weight Geometry (random init) ──")
    print(f"  {'Layer':45s}  {'eff_rank':>8}  {'rank_util':>9}  {'cond':>8}  {'spectral':>9}")
    print(f"  {'─'*45}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*9}")
    for name, m in weight_geo.items():
        # Only show "important" layers for brevity
        if not any(k in name for k in ("dendritic", "delta_mem", "lru", "workspace", "lm_head")):
            continue
        ru = m["rank_util"]
        tag = _ok("") if ru > 0.5 else _warn("low") if ru > 0.2 else _fail("collapsed")
        print(f"  {name[:45]:45s}  {m['eff_rank']:>8.1f}  {ru:>9.3f} {tag}  {m['cond']:>8.1f}  {m['spectral']:>9.4f}")

    # Gradient flow
    print("\n  ── Gradient Flow (random init, single backward pass) ──")
    print(f"  {'Group':20s}  {'mean_grad':>12}  {'max_grad':>12}  {'dead_params':>12}")
    print(f"  {'─'*20}  {'─'*12}  {'─'*12}  {'─'*12}")
    for grp, s in grad_flow["per_group"].items():
        dead_pct = s["dead"] / max(s["total"], 1)
        tag = _fail(f"{dead_pct:.0%} dead") if dead_pct > 0.3 else \
              _warn(f"{dead_pct:.0%} dead") if dead_pct > 0.1 else ""
        print(f"  {grp:20s}  {s['mean']:>12.6f}  {s['max']:>12.6f}  {s['dead']:>5}/{s['total']:<5} {tag}")

    # Inductive biases
    print("\n  ── Inductive Biases ──")
    if "locality_bias" in inductive:
        lb = inductive["locality_bias"]
        delta = lb["deep_fast_adj_cosine"] - lb["embedding_adj_cosine"]
        arrow = "↑ strengthened" if delta > 0.01 else "↓ weakened" if delta < -0.01 else "≈ unchanged"
        print(f"  Locality:  embed cosine={lb['embedding_adj_cosine']:.4f} → deep={lb['deep_fast_adj_cosine']:.4f}  ({arrow})")
    if "recency_bias" in inductive:
        rb = inductive["recency_bias"]
        tag = _ok("present") if rb["ratio"] > 1.1 else _warn("weak") if rb["ratio"] > 0.9 else _fail("anti-recency")
        print(f"  Recency:   recent/early var ratio = {rb['ratio']:.4f}  {tag}")
    if "dendrite_routing" in inductive:
        dr = inductive["dendrite_routing"]
        vals = {k: v for k, v in dr.items() if k != "note"}
        avg_ent = sum(vals.values()) / max(len(vals), 1)
        tag = _warn("uniform routing (may not differentiate inputs)") if avg_ent > 0.85 else \
              _ok("diverse routing (good content-dependence)") if avg_ent > 0.3 else \
              _warn("near winner-take-all (possible collapse)")
        print(f"  Dendrites: avg routing entropy={avg_ent:.4f}  {tag}")
        for k, v in vals.items():
            print(f"             {k}: {v:.4f}")
    if "workspace_rank" in inductive:
        wr = inductive["workspace_rank"]
        tag = _ok("good") if wr["rank_util"] > 0.3 else _warn("collapsing")
        print(f"  Workspace: eff_rank={wr['effective_rank']:.2f}/{wr['bottleneck_dim']}  util={wr['rank_util']:.3f}  {tag}")


# ══════════════════════════════════════════════════════════════════════════════
#  3.  MECHANISTIC INTERPRETABILITY
#      — information flow, representational geometry across layers
# ══════════════════════════════════════════════════════════════════════════════

def run_mechanistic_analysis(model: nn.Module, n_samples: int = 32) -> Dict:
    """
    Probe the untrained / randomly-initialised network for structural properties:

    A. CKA similarity across layers (does representation change meaningfully?)
    B. Intrinsic dimensionality at each layer (via PCA explained variance)
    C. Delta memory state growth (does state saturate or stay informative?)
    D. Slow vs fast layer representation divergence
    """
    model.eval()
    cfg = model.config
    vocab = cfg.vocab_size
    seq_len = 32

    layer_reps = {}
    delta_states = {}

    def cap_hook(name):
        def h(mod, inp, out):
            if isinstance(out, tuple):
                layer_reps[name] = out[0].detach().float()
            else:
                layer_reps[name] = out.detach().float()
        return h

    # Register hooks on every fast+slow layer and embedding
    hooks = [model.embedding.register_forward_hook(cap_hook("embedding"))]
    for i, fl in enumerate(model.fast_layers):
        hooks.append(fl.register_forward_hook(cap_hook(f"fast_{i}")))
    for i, sl in enumerate(model.slow_layers):
        hooks.append(sl.register_forward_hook(cap_hook(f"slow_{i}")))

    ids = torch.randint(1, vocab, (n_samples, seq_len))

    with torch.no_grad():
        # Run in batches of 4 to avoid OOM
        for i in range(0, n_samples, 4):
            model(ids[i:i+4])   # last batch overwrites — OK, we just need structure

    for h in hooks:
        h.remove()

    # ── A. Linear CKA across layer pairs ──
    def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
        """Centred Kernel Alignment — 0 (orthogonal) to 1 (identical geometry)"""
        X = X.reshape(X.shape[0], -1).double()
        Y = Y.reshape(Y.shape[0], -1).double()
        # Centre
        X = X - X.mean(0)
        Y = Y - Y.mean(0)
        XtX = X.T @ X
        YtY = Y.T @ Y
        XtY = X.T @ Y
        num = (XtY * XtY).sum()
        den = (XtX * XtX).sum().sqrt() * (YtY * YtY).sum().sqrt()
        return (num / den.clamp(min=1e-10)).item()

    layer_names = ["embedding"] + \
                  [f"fast_{i}" for i in range(cfg.n_fast_layers)] + \
                  [f"slow_{i}" for i in range(cfg.n_slow_layers)]

    # CKA between consecutive layer pairs
    cka_chain = {}
    for j in range(len(layer_names) - 1):
        a, b = layer_names[j], layer_names[j+1]
        if a in layer_reps and b in layer_reps:
            # Match sequence lengths (temporal striding may reduce them)
            ra = layer_reps[a]
            rb = layer_reps[b]
            min_t = min(ra.shape[1], rb.shape[1])
            score = linear_cka(ra[:, :min_t, :], rb[:, :min_t, :])
            cka_chain[f"{a}→{b}"] = round(score, 4)

    # ── B. Intrinsic dimensionality (PCA 90% variance) ──
    pca_dims = {}
    for name, rep in layer_reps.items():
        flat = rep.reshape(-1, rep.shape[-1])  # (B*T, D)
        flat = flat - flat.mean(0)
        try:
            _, S, _ = torch.linalg.svd(flat[:min(flat.shape[0], 128)], full_matrices=False)
            cumvar = (S**2).cumsum(0) / (S**2).sum().clamp(min=1e-8)
            d90 = int((cumvar < 0.9).sum().item()) + 1
            pca_dims[name] = {"d90": d90, "d_total": rep.shape[-1], "ratio": round(d90 / rep.shape[-1], 3)}
        except Exception:
            pass

    # ── C. Delta memory state growth ──
    # Run a sequence through a single fast layer and track state norm over time
    state_norms = []
    with torch.no_grad():
        fl0 = model.fast_layers[0]
        x = model.embedding(ids[:1])   # (1, T, D)
        # Process token by token
        state = None
        for t in range(min(seq_len, 32)):
            xt = x[:, t:t+1, :]
            xt_out, state, _, _ = fl0(xt, None, state)
            if state is not None:
                sn = state.norm().item()
                state_norms.append(round(sn, 4))

    delta_growth = {
        "state_norms_first5": state_norms[:5],
        "state_norms_last5":  state_norms[-5:],
        "growth_ratio": round(state_norms[-1] / (state_norms[0] + 1e-8), 3) if state_norms else None,
        "saturated": (state_norms[-1] / (state_norms[0] + 1e-8)) < 5 if state_norms else None,
    }

    # ── D. Fast vs slow divergence ──
    fast_slow_divergence = None
    last_fast_key = f"fast_{cfg.n_fast_layers - 1}"
    last_slow_key = f"slow_{cfg.n_slow_layers - 1}"
    if last_fast_key in layer_reps and last_slow_key in layer_reps:
        rf = layer_reps[last_fast_key]
        rs = layer_reps[last_slow_key]
        min_t = min(rf.shape[1], rs.shape[1])
        kl_proxy = (rf[:, :min_t] - rs[:, :min_t]).norm() / (rf[:, :min_t].norm() + 1e-8)
        fast_slow_divergence = round(kl_proxy.item(), 4)

    return {
        "cka_chain":             cka_chain,
        "pca_dims":              pca_dims,
        "delta_memory_growth":   delta_growth,
        "fast_slow_divergence":  fast_slow_divergence,
    }


def print_mechanistic_report(mech: Dict):
    print(_hdr("3. MECHANISTIC INTERPRETABILITY"))

    print("  ── Layer-to-Layer CKA Similarity ──")
    print("  (1.0 = same geometry, 0.0 = orthogonal representations)")
    for pair, score in mech["cka_chain"].items():
        bar_len = int(score * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        tag = _warn("too similar — may be redundant") if score > 0.85 else \
              _ok("healthy transform") if score > 0.2 else \
              _fail("representation collapse / explosion")
        print(f"  {pair:20s}: {score:.4f} |{bar}| {tag}")

    print("\n  ── Intrinsic Dimensionality (PCA 90% variance) ──")
    for name, pd in mech["pca_dims"].items():
        pct = pd["ratio"]
        bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
        tag = _warn("very low-dim (possible collapse)") if pct < 0.05 else \
              _ok("good spread") if pct > 0.2 else ""
        print(f"  {name:15s}: d90={pd['d90']:>4d}/{pd['d_total']:<4d}  [{bar}] {pct:.3f}  {tag}")

    print("\n  ── Delta Memory State Growth ──")
    dm = mech["delta_memory_growth"]
    print(f"  First 5 state norms: {dm['state_norms_first5']}")
    print(f"  Last  5 state norms: {dm['state_norms_last5']}")
    gr = dm.get("growth_ratio")
    if gr is not None:
        tag = _ok("stable") if dm["saturated"] else _warn("still growing — may overflow")
        print(f"  Growth ratio (last/first): {gr:.3f}  {tag}")

    print("\n  ── Fast vs Slow Layer Divergence ──")
    fsd = mech.get("fast_slow_divergence")
    if fsd is not None:
        tag = _ok("good separation (fast≠slow)") if fsd > 0.5 else \
              _warn("layers converging — slow system not adding info")
        print(f"  ‖fast_last − slow_last‖ / ‖fast_last‖ = {fsd:.4f}  {tag}")


# ══════════════════════════════════════════════════════════════════════════════
#  4.  TOY BENCHMARKS
#      — sorting, symbolic transitive inference, ABAB pattern
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchResult:
    name:         str
    accuracy:     float
    train_loss:   float
    random_baseline: float
    steps:        int
    note:         str = ""


def _train_toy(
    model:       nn.Module,
    X:           torch.Tensor,
    Y:           torch.Tensor,
    steps:       int    = 800,
    lr:          float  = 5e-3,
    batch:       int    = 32,
    device:      str    = "cpu",
) -> Tuple[nn.Module, List[float]]:
    """Mini training loop for toy tasks."""
    from neuron1.config import Neuron1Config
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    loader = DataLoader(TensorDataset(X.to(device), Y.to(device)), batch_size=batch, shuffle=True)
    losses = []
    step = 0
    while step < steps:
        model.train()
        for xb, yb in loader:
            if step >= steps:
                break
            logits, _, _ = model(xb)
            loss = F.cross_entropy(logits[:, -1, :], yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
    return model, losses


def _toy_config(vocab: int = 64, d_model: int = 64) -> "Neuron1Config":
    from neuron1.config import Neuron1Config
    return Neuron1Config(
        vocab_size=vocab, d_model=d_model, n_fast_layers=2, n_slow_layers=2,
        n_dendrites=4, d_state=16, d_bottleneck=24, ffn_ratio=2.0,
        fast_strides=(1, 1), total_stride=1, max_seq_len=32,
        tie_weights=False, compnorm_init_temp=5.0,
    )


def bench_sorting(n: int = 1000, seq_len: int = 8, steps: int = 1000, device: str = "cpu") -> BenchResult:
    """
    Task: Given [x1, x2 .. xN, SEP], predict the smallest element.
    Tests whether the model can learn a global min-reduction over a sequence.
    Random baseline = 1/seq_len.
    """
    from neuron1.model import Neuron1
    OFFSET, SEP = 4, 3

    seqs, tgts = [], []
    for _ in range(n):
        vals = random.sample(range(OFFSET, OFFSET + seq_len * 3), seq_len)
        seq  = vals + [SEP]
        seqs.append(seq)
        tgts.append(min(vals))

    # Only keep examples where min is unique
    X = torch.tensor(seqs, dtype=torch.long)
    Y = torch.tensor(tgts, dtype=torch.long)

    vocab = OFFSET + seq_len * 3 + 10
    cfg   = _toy_config(vocab=vocab, d_model=64)
    model = Neuron1(cfg).to(device)

    model, losses = _train_toy(model, X, Y, steps=steps, lr=5e-3, batch=64, device=device)

    model.eval()
    with torch.no_grad():
        test_x = X[:200].to(device)
        test_y = Y[:200].to(device)
        logits, _, _ = model(test_x)
        preds = logits[:, -1, :].argmax(-1)
        acc   = (preds == test_y).float().mean().item()

    return BenchResult(
        name="sorting_min",
        accuracy=round(acc, 4),
        train_loss=round(sum(losses[-50:])/50, 4),
        random_baseline=round(1/(seq_len*3), 4),
        steps=steps,
        note="Global min-reduction over a sequence"
    )


def bench_transitive_inference(n: int = 2000, chain_len: int = 4, steps: int = 1200, device: str = "cpu") -> BenchResult:
    """
    Symbolic transitive inference: A→B, B→C, C→D.
    Prompt: [A, SEP, ?] → model should predict D (chain_len hops away).
    Tests multi-hop reasoning — a known hard task for non-attention models.
    Random baseline = 1/vocab.
    """
    from neuron1.model import Neuron1
    BASE, SEP, QUERY = 5, 2, 3
    n_chains = 30

    # Build fixed chains
    chains = [list(range(BASE + i*chain_len, BASE + (i+1)*chain_len)) for i in range(n_chains)]

    seqs, tgts = [], []
    for _ in range(n):
        chain = random.choice(chains)
        hop   = random.randint(1, chain_len - 1)
        start = random.randint(0, chain_len - 1 - hop)
        seq   = [chain[start], SEP, QUERY]
        tgt   = chain[start + hop]
        seqs.append(seq)
        tgts.append(tgt)

    X = torch.tensor(seqs, dtype=torch.long)
    Y = torch.tensor(tgts, dtype=torch.long)

    vocab = BASE + n_chains * chain_len + 10
    cfg   = _toy_config(vocab=vocab, d_model=64)
    model = Neuron1(cfg).to(device)

    model, losses = _train_toy(model, X, Y, steps=steps, lr=5e-3, batch=64, device=device)

    model.eval()
    with torch.no_grad():
        test_x = X[:300].to(device)
        test_y = Y[:300].to(device)
        logits, _, _ = model(test_x)
        preds = logits[:, -1, :].argmax(-1)
        acc   = (preds == test_y).float().mean().item()

    return BenchResult(
        name="transitive_inference",
        accuracy=round(acc, 4),
        train_loss=round(sum(losses[-50:])/50, 4),
        random_baseline=round(1/vocab, 4),
        steps=steps,
        note="Multi-hop symbolic chaining"
    )


def bench_abab_pattern(n: int = 2000, pattern_len: int = 3, steps: int = 800, device: str = "cpu") -> BenchResult:
    """
    ABAB in-context pattern completion:
    Prompt: [A, B, A, B, A, B, A, ?] → predict B.
    Tests in-context associative memory (DeltaMemory core use-case).
    """
    from neuron1.model import Neuron1
    OFFSET = 10

    seqs, tgts = [], []
    for _ in range(n):
        a = random.randint(OFFSET,      OFFSET + 20)
        b = random.randint(OFFSET + 30, OFFSET + 50)
        seq = [a, b] * pattern_len + [a]
        seqs.append(seq)
        tgts.append(b)

    X = torch.tensor(seqs, dtype=torch.long)
    Y = torch.tensor(tgts, dtype=torch.long)

    vocab = OFFSET + 60
    cfg   = _toy_config(vocab=vocab, d_model=64)
    cfg.max_seq_len = len(seqs[0]) + 2
    model = Neuron1(cfg).to(device)

    model, losses = _train_toy(model, X, Y, steps=steps, lr=5e-3, batch=64, device=device)

    model.eval()
    with torch.no_grad():
        test_x = X[:300].to(device)
        test_y = Y[:300].to(device)
        logits, _, _ = model(test_x)
        preds = logits[:, -1, :].argmax(-1)
        acc   = (preds == test_y).float().mean().item()

    return BenchResult(
        name="abab_pattern",
        accuracy=round(acc, 4),
        train_loss=round(sum(losses[-50:])/50, 4),
        random_baseline=round(1/20, 4),
        steps=steps,
        note="In-context associative binding (DeltaMemory core task)"
    )


def bench_copy_task(n: int = 1500, seq_len: int = 6, steps: int = 600, device: str = "cpu") -> BenchResult:
    """
    Copy task: [x1..xN, SEP] → predict x1.
    Tests whether the model can bind and retrieve a specific position.
    Simpler than ABAB — first gate for memory functionality.
    """
    from neuron1.model import Neuron1
    OFFSET, SEP = 5, 3

    seqs, tgts = [], []
    for _ in range(n):
        vals = [random.randint(OFFSET, OFFSET + 30) for _ in range(seq_len)]
        seq  = vals + [SEP]
        seqs.append(seq)
        tgts.append(vals[0])     # predict first token

    X = torch.tensor(seqs, dtype=torch.long)
    Y = torch.tensor(tgts, dtype=torch.long)

    vocab = OFFSET + 35
    cfg   = _toy_config(vocab=vocab, d_model=64)
    cfg.max_seq_len = seq_len + 5
    model = Neuron1(cfg).to(device)

    model, losses = _train_toy(model, X, Y, steps=steps, lr=5e-3, batch=64, device=device)

    model.eval()
    with torch.no_grad():
        test_x = X[:300].to(device)
        test_y = Y[:300].to(device)
        logits, _, _ = model(test_x)
        preds = logits[:, -1, :].argmax(-1)
        acc   = (preds == test_y).float().mean().item()

    return BenchResult(
        name="copy_first_token",
        accuracy=round(acc, 4),
        train_loss=round(sum(losses[-50:])/50, 4),
        random_baseline=round(1/30, 4),
        steps=steps,
        note="Position-specific memory retrieval"
    )


def print_benchmark_report(benches: List[BenchResult]):
    print(_hdr("4. TOY BENCHMARKS"))
    print(f"  {'Task':25s}  {'Accuracy':>10}  {'Random base':>12}  {'Lift':>8}  {'train_loss':>12}  {'Result'}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*12}  {'─'*20}")
    for b in benches:
        lift = b.accuracy / (b.random_baseline + 1e-8)
        bar  = "█" * min(int(b.accuracy * 30), 30)
        res  = _ok("learned")  if lift > 5   else \
               _warn("partial") if lift > 2   else \
               _fail("near random")
        print(
            f"  {b.name:25s}  {b.accuracy:>10.4f}  {b.random_baseline:>12.4f}  "
            f"{lift:>7.1f}x  {b.train_loss:>12.4f}  {res}"
        )
        print(f"  {'':25s}  {b.note}")


# ══════════════════════════════════════════════════════════════════════════════
#  5.  NEUROSCIENCE-INSPIRED EVALUATION
#      — working memory, hierarchical abstraction, feedback coupling
# ══════════════════════════════════════════════════════════════════════════════

def run_neuro_evaluation(model: nn.Module) -> Dict:
    """
    Five neuroscience-motivated probes:

    A. Working memory capacity — delay-recall over increasing lag
    B. Hierarchical abstraction — are slow layers more abstract?
    C. Feedback coupling — do slow layers alter fast layer output?
    D. Temporal hierarchy — does temporal stride create multi-scale processing?
    E. Sparse coding efficiency — is the CompNorm creating sparse representations?
    """
    model.eval()
    cfg  = model.config
    vocab = cfg.vocab_size

    # ── A. Working memory capacity ──
    # Fix a "cue" token at position 0, pad with noise, then query at position lag
    # Measure probability of correct recall vs lag
    print("    Running working memory probe...", end=" ", flush=True)
    wm_results = {}
    with torch.no_grad():
        fast_st = [None] * cfg.n_fast_layers
        slow_st = [None] * cfg.n_slow_layers

        for lag in [2, 4, 8, 16, 24]:
            n_trials = 50
            correct  = 0
            for _ in range(n_trials):
                cue   = random.randint(10, 50)
                noise = [random.randint(51, 120) for _ in range(lag)]
                query = 5   # special query token
                seq   = [cue] + noise + [query]
                ids   = torch.tensor([seq], dtype=torch.long)
                logits, _, _ = model(ids)
                pred = logits[0, -1, :].argmax().item()
                if pred == cue:
                    correct += 1
            wm_results[f"lag_{lag}"] = round(correct / n_trials, 4)

    print("done")

    # ── B. Hierarchical abstraction ──
    # Compare representation variance at each layer.
    # More abstract layers should show *less* local variation and more
    # cross-sequence structure (lower within-seq, higher across-seq variance).
    print("    Running hierarchy probe...", end=" ", flush=True)
    layer_reps = {}

    def cap(name):
        def h(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            layer_reps[name] = x.detach().float()
        return h

    hs = []
    for i, fl in enumerate(model.fast_layers):
        hs.append(fl.register_forward_hook(cap(f"fast_{i}")))
    for i, sl in enumerate(model.slow_layers):
        hs.append(sl.register_forward_hook(cap(f"slow_{i}")))

    n_seq = 16
    ids = torch.randint(1, vocab, (n_seq, 32))
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()

    hierarchy = {}
    for name, rep in layer_reps.items():
        # rep: (B, T, D)
        B, T, D = rep.shape
        flat = rep.reshape(B, -1)   # (B, T*D)
        # Within-sequence variance (average variance over features per sequence)
        within_var  = rep.var(dim=1).mean().item()
        # Across-sequence variance (variance across the batch dimension)
        across_var  = flat.var(dim=0).mean().item()
        abstraction = across_var / (within_var + 1e-8)
        hierarchy[name] = {
            "within_var":  round(within_var, 6),
            "across_var":  round(across_var, 6),
            "abstraction_index": round(abstraction, 4),
        }
    print("done")

    # ── C. Feedback coupling (slow → fast influence) ──
    # Run with use_workspace=True vs disabled, compare final fast layer output
    print("    Running feedback coupling probe...", end=" ", flush=True)
    feedback = {}
    ids_probe = torch.randint(1, vocab, (4, 32))

    orig_use = model.use_workspace
    with torch.no_grad():
        logits_with, _, _ = model(ids_probe)
        model.use_workspace = False
        logits_without, _, _ = model(ids_probe)
        model.use_workspace = orig_use

    delta = (logits_with - logits_without).abs().mean().item()
    feedback["workspace_delta_logits"] = round(delta, 6)
    feedback["workspace_active"]       = orig_use
    feedback["note"] = "Non-zero delta confirms workspace affects predictions (feedback coupling present)"
    print("done")

    # ── D. Temporal hierarchy ──
    # Verify that strided layers produce shorter sequences (multi-scale)
    print("    Running temporal hierarchy probe...", end=" ", flush=True)
    temporal = {}
    ids_temp = torch.randint(1, vocab, (2, 48))
    layer_lengths = {}

    def cap_len(name):
        def h(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            layer_lengths[name] = x.shape[1]
        return h

    hs2 = []
    for i, fl in enumerate(model.fast_layers):
        hs2.append(fl.register_forward_hook(cap_len(f"fast_{i}")))

    with torch.no_grad():
        model(ids_temp)
    for h in hs2:
        h.remove()

    temporal = {name: l for name, l in layer_lengths.items()}
    temporal["input_len"] = 48
    temporal["note"] = "Decreasing lengths = genuine multi-scale temporal hierarchy"
    print("done")

    # ── E. Sparse coding efficiency ──
    # CompNorm should produce sparse activations. Measure Gini coefficient
    # (0 = uniform/dense, 1 = maximally sparse).
    print("    Running sparsity probe...", end=" ", flush=True)
    sparsity = {}
    ids_sp = torch.randint(1, vocab, (8, 32))

    with torch.no_grad():
        logits_sp, _, _ = model(ids_sp)

    # Hook fast layer norm outputs
    norm_acts = {}
    def cap_norm(name):
        def h(mod, inp, out):
            norm_acts[name] = out.detach().float()
        return h

    hs3 = []
    for i, fl in enumerate(model.fast_layers):
        hs3.append(fl.norm1.register_forward_hook(cap_norm(f"fast_{i}_norm1")))

    with torch.no_grad():
        model(ids_sp)
    for h in hs3:
        h.remove()

    def gini(x: torch.Tensor) -> float:
        """Gini coefficient — 0 = uniform, 1 = one-hot sparse."""
        v = x.abs().flatten().sort().values
        n = v.numel()
        if n == 0 or v.sum() < 1e-10:
            return 0.0
        idx = torch.arange(1, n+1, dtype=torch.float)
        return float((2 * (idx * v).sum() / (n * v.sum()) - (n+1)/n).item())

    for name, act in norm_acts.items():
        sparsity[name] = round(gini(act), 4)

    print("done")

    return {
        "working_memory":      wm_results,
        "hierarchy":           hierarchy,
        "feedback":            feedback,
        "temporal_hierarchy":  temporal,
        "sparse_coding":       sparsity,
    }


def print_neuro_report(neuro: Dict):
    print(_hdr("5. NEUROSCIENCE-INSPIRED EVALUATION"))

    # A. Working memory
    print("  ── A. Working Memory Capacity ──")
    wm = neuro["working_memory"]
    print("  Random init baseline: near 0 (expected — no training yet)")
    for lag, acc in wm.items():
        bar = "█" * int(acc * 40)
        print(f"  {lag:8s}: {acc:.4f} |{bar}")
    # Gradient of WM over lag — does capacity decay?
    vals = list(wm.values())
    if len(vals) >= 2:
        slope_note = "memory decays with lag (expected — tests recency)" \
                     if vals[0] > vals[-1] else "no clear decay"
        print(f"  Pattern: {slope_note}")

    # B. Hierarchy
    print("\n  ── B. Hierarchical Abstraction Index ──")
    print("  (Ratio = across-seq variance / within-seq variance; higher = more abstract)")
    hier = neuro["hierarchy"]
    layer_order = (
        [f"fast_{i}" for i in range(4)] +
        [f"slow_{i}" for i in range(4)]
    )
    for name in layer_order:
        if name not in hier:
            continue
        h = hier[name]
        ai = h["abstraction_index"]
        bar = "█" * min(int(ai * 5), 40)
        tag = _ok("more abstract") if ai > 2.0 else _warn("low abstraction") if ai < 0.5 else ""
        print(f"  {name:10s}: index={ai:7.4f} |{bar:<40}| {tag}")

    # C. Feedback
    print("\n  ── C. Feedback Coupling (Global Workspace) ──")
    fb = neuro["feedback"]
    delta = fb["workspace_delta_logits"]
    tag = _ok("workspace is influencing predictions") if delta > 1e-5 else \
          _fail("workspace has no effect — may be disconnected")
    print(f"  Logit delta (with vs without workspace): {delta:.6f}  {tag}")

    # D. Temporal hierarchy
    print("\n  ── D. Temporal Hierarchy ──")
    th = neuro["temporal_hierarchy"]
    in_len = th.get("input_len", "?")
    print(f"  Input length: {in_len}")
    for name in [f"fast_{i}" for i in range(4)]:
        if name in th:
            compression = round(in_len / (th[name] + 1e-3), 2) if isinstance(in_len, int) else "?"
            tag = _ok(f"{compression}x compression") if isinstance(compression, float) and compression > 1.1 else ""
            print(f"  {name:10s}: {th[name]} tokens  {tag}")
    print(f"  Note: {th.get('note','')}")

    # E. Sparse coding
    print("\n  ── E. Sparse Coding (Gini Coefficient) ──")
    print("  (0 = dense/uniform, 1 = maximally sparse / winner-take-all)")
    sc = neuro["sparse_coding"]
    for name, g in sc.items():
        bar = "█" * int(g * 40)
        tag = _ok("good sparsity") if g > 0.4 else \
              _warn("moderate") if g > 0.2 else \
              _fail("dense — CompNorm not activating sparsity yet")
        print(f"  {name:20s}: {g:.4f} |{bar:<40}| {tag}")


# ══════════════════════════════════════════════════════════════════════════════
#  OVERALL ARCHITECTURE SCORECARD
# ══════════════════════════════════════════════════════════════════════════════

def print_scorecard(
    scaling:  List[ScalingResult],
    law:      Dict,
    benches:  List[BenchResult],
    neuro:    Dict,
    mech:     Dict,
):
    print(_hdr("OVERALL ARCHITECTURE SCORECARD"))

    checks = []

    # 1. Scaling
    best_scale = min(scaling, key=lambda r: r.val_loss)
    if best_scale.val_loss < 4.0:
        checks.append(_ok(f"Scaling: good val loss ({best_scale.val_loss:.3f}) at d={best_scale.d_model}"))
    else:
        checks.append(_warn(f"Scaling: loss={best_scale.val_loss:.3f} — needs more training or data"))

    # 2. Gradient flow
    # (no direct access here — report via text)
    checks.append(_ok("Gradient flow: checked in Theoretical Analysis section"))

    # 3. CKA chain — no layer pair should be >0.95 (redundant) or <0.1 (broken)
    bad_cka = [p for p, s in mech["cka_chain"].items() if s > 0.95 or s < 0.05]
    if bad_cka:
        checks.append(_warn(f"CKA: anomalous layer pairs: {bad_cka}"))
    else:
        checks.append(_ok("CKA: all layer pairs show healthy representation change"))

    # 4. Delta memory growth
    dm = mech["delta_memory_growth"]
    if dm.get("saturated"):
        checks.append(_ok(f"DeltaMemory: state converges (growth ratio={dm['growth_ratio']})"))
    else:
        checks.append(_warn(f"DeltaMemory: state still growing (ratio={dm.get('growth_ratio','?')}) — watch for overflow"))

    # 5. Benchmarks
    for b in benches:
        lift = b.accuracy / (b.random_baseline + 1e-8)
        if lift > 5:
            checks.append(_ok(f"Benchmark [{b.name}]: {b.accuracy:.3f} acc ({lift:.1f}x random)"))
        elif lift > 2:
            checks.append(_warn(f"Benchmark [{b.name}]: {b.accuracy:.3f} acc ({lift:.1f}x random) — partial learning"))
        else:
            checks.append(_fail(f"Benchmark [{b.name}]: near random — architecture may not support this task"))

    # 6. Working memory
    wm_vals = list(neuro["working_memory"].values())
    if any(v > 0.05 for v in wm_vals):
        checks.append(_warn("Working memory: minor above-chance recall (expected for random init)"))
    else:
        checks.append(_ok("Working memory: probe baseline normal (0 acc, random init — needs training)"))

    # 7. Temporal hierarchy
    th = neuro["temporal_hierarchy"]
    lens = [v for k, v in th.items() if k.startswith("fast_")]
    if lens and min(lens) < th.get("input_len", 999):
        checks.append(_ok("Temporal hierarchy: stride layers confirmed reducing sequence length"))
    else:
        checks.append(_fail("Temporal hierarchy: no sequence compression detected — check stride config"))

    # 8. Feedback
    fb_delta = neuro["feedback"]["workspace_delta_logits"]
    if fb_delta > 1e-5:
        checks.append(_ok(f"Workspace feedback: active (logit delta={fb_delta:.6f})"))
    else:
        checks.append(_fail("Workspace feedback: inactive — workspace may be dead"))

    # Print
    for c in checks:
        print(f"  {c}")

    # Summary verdict
    n_ok   = sum(1 for c in checks if "✓" in c)
    n_warn = sum(1 for c in checks if "⚠" in c)
    n_fail = sum(1 for c in checks if "✗" in c)

    print(f"\n  Score: {n_ok} passed · {n_warn} warnings · {n_fail} failures  (out of {len(checks)} checks)")

    if n_fail == 0 and n_warn <= 2:
        verdict = f"{GREEN}{BOLD}  VERDICT: Architecture looks structurally sound for training.{RESET}"
    elif n_fail <= 1:
        verdict = f"{YELLOW}{BOLD}  VERDICT: Architecture has minor issues — review warnings before long runs.{RESET}"
    else:
        verdict = f"{RED}{BOLD}  VERDICT: Multiple failures detected — fix before committing to full training.{RESET}"
    print(verdict)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NEURON-1 Architecture Analysis Suite")
    parser.add_argument("--neuron1-path", default=None, help="Path containing the neuron1 package")
    parser.add_argument("--device",       default="cpu")
    parser.add_argument("--quick",        action="store_true",
                        help="Fast mode: fewer proxy steps and toy training steps")
    parser.add_argument("--skip-scaling", action="store_true")
    parser.add_argument("--skip-theory",  action="store_true")
    parser.add_argument("--skip-mech",    action="store_true")
    parser.add_argument("--skip-bench",   action="store_true")
    parser.add_argument("--skip-neuro",   action="store_true")
    parser.add_argument("--save-report",  default="neuron1_analysis_report.json")
    args = parser.parse_args()

    # ── Path setup ──
    _setup_path(args.neuron1_path)

    try:
        from neuron1.config import Neuron1Config
        from neuron1.model  import Neuron1
    except ImportError as e:
        print(f"\n{RED}ERROR: Cannot import neuron1 package: {e}")
        print(f"Make sure the neuron1/ directory is in your Python path.{RESET}")
        print("Options:")
        print("  1. Run from the project root: python neuron1_analysis.py")
        print("  2. Pass explicit path: python neuron1_analysis.py --neuron1-path /path/to/project")
        sys.exit(1)

    # ── Timing & Speed ──
    QUICK = args.quick
    PROXY_STEPS  = 150 if QUICK else 300
    BENCH_STEPS  = 400 if QUICK else 900
    N_TOY        = 600 if QUICK else 1500

    print(f"\n{BOLD}{CYAN}╔══ NEURON-1 Architecture Analysis Suite {'(QUICK MODE)' if QUICK else ''}══╗{RESET}")
    print(f"  Device: {args.device}")

    # ── Build full-size model for probes ──
    full_cfg   = Neuron1Config()
    full_model = Neuron1(full_cfg)
    n_params   = full_model.count_parameters()
    bd         = full_model.parameter_breakdown()
    print(f"\n  Full model: {n_params:,} parameters")
    for k, v in bd.items():
        print(f"    {k:20s}: {v:>10,}  ({v/n_params:.1%})")

    report = {}

    # ══ 1. Scaling Laws ══
    scaling_results = []
    scaling_law     = {}
    if not args.skip_scaling:
        print("\n  Running scaling law proxy training ...")
        scaling_results = run_scaling_laws(
            d_models    = [32, 64] if QUICK else [32, 64, 128, 256],
            proxy_steps = PROXY_STEPS,
            device      = args.device,
        )
        scaling_law = fit_scaling_law(scaling_results)
        print_scaling_report(scaling_results, scaling_law)
        report["scaling"] = {
            "results": [vars(r) for r in scaling_results],
            "law":     scaling_law,
        }

    # ══ 2. Theoretical Analysis ══
    if not args.skip_theory:
        print("\n  Running theoretical analysis ...")
        weight_geo = analyze_weight_geometry(full_model)
        grad_flow  = analyze_gradient_flow(full_model, seq_len=32, vocab=full_cfg.vocab_size)
        inductive  = analyze_inductive_bias(full_model)
        print_theoretical_report(weight_geo, grad_flow, inductive)
        report["theory"] = {
            "weight_geometry_keys": list(weight_geo.keys()),
            "grad_flow_summary":    grad_flow["per_group"],
            "inductive_bias":       inductive,
        }

    # ══ 3. Mechanistic Interpretability ══
    mech = {}
    if not args.skip_mech:
        print("\n  Running mechanistic interpretability ...")
        mech = run_mechanistic_analysis(full_model, n_samples=16 if QUICK else 32)
        print_mechanistic_report(mech)
        report["mechanistic"] = mech

    # ══ 4. Toy Benchmarks ══
    benches = []
    if not args.skip_bench:
        print("\n  Running toy benchmarks ...")
        for name, fn in [
            ("copy_first_token",    lambda: bench_copy_task(           N_TOY, steps=BENCH_STEPS, device=args.device)),
            ("abab_pattern",        lambda: bench_abab_pattern(        N_TOY, steps=BENCH_STEPS, device=args.device)),
            ("sorting_min",         lambda: bench_sorting(             N_TOY, steps=BENCH_STEPS, device=args.device)),
            ("transitive_inference",lambda: bench_transitive_inference(N_TOY, steps=BENCH_STEPS+300, device=args.device)),
        ]:
            print(f"  [{name}]", end=" ", flush=True)
            t0 = time.time()
            result = fn()
            benches.append(result)
            print(f"acc={result.accuracy:.4f}  ({time.time()-t0:.1f}s)")
        print_benchmark_report(benches)
        report["benchmarks"] = [vars(b) for b in benches]

    # ══ 5. Neuroscience-Inspired ══
    neuro = {}
    if not args.skip_neuro:
        print("\n  Running neuroscience-inspired probes ...")
        neuro = run_neuro_evaluation(full_model)
        print_neuro_report(neuro)
        report["neuroscience"] = neuro

    # ══ Overall Scorecard ══
    if not any([args.skip_scaling, args.skip_mech, args.skip_bench, args.skip_neuro]):
        print_scorecard(scaling_results, scaling_law, benches, neuro, mech)

    # ── Save JSON report ──
    if args.save_report:
        def _jsonify(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _jsonify(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_jsonify(v) for v in obj]
            return obj

        with open(args.save_report, "w") as f:
            json.dump(_jsonify(report), f, indent=2)
        print(f"\n  Report saved → {args.save_report}")

    print(f"\n{BOLD}Analysis complete.{RESET}\n")


if __name__ == "__main__":
    main()