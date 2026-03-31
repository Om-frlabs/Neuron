# NEURON-1: Hybrid Sparse MoE Architecture

> A ~20.15M parameter Hybrid SSM-Attention architecture modernized for 2026 frontier standards.

## Architecture Overview

NEURON-1 is a high-performance hybrid model that integrates **Sparse Mixture of Experts (MoE)** routing, **Multi-head Latent Attention (MLA)** memory compression, and **Hardware-Aware Parallel Scans (SSD)**. It is designed to overcome the quadratic scaling limitations of pure Transformers while maintaining exact associative recall.

### High-Level Topology
```text
Input → Embedding 
  → 4x Fast Layers (stride=2 sum): 
      Parallel SSD Scan (Decayed Linear Attention) → Sparse MoE (Top-2 / 8 Experts)
  → Global Workspace (MLA Bottleneck):
      Latent Compression (h_mem → c_t) → KV-Cache Expansion → Cross-Attention
  → Hybrid Attention (Full Causal Recall @ T/4 resolution)
  → TemporalUpsample (T/4 → T via Learned interpolation + Gated skip)
  → 4x Slow Layers: 
      GatedLRU (Recurrent Backbone) → Sparse MoE (Top-2 / 8 Experts)
  → Final Norm → LM Head
```

## Key Innovations (Phase 7-9 Modernization)

| Component | Technology | Benefit |
|---|---|---|
| **Sparse MoE** | Top-2 / 8 Expert Routing | 20M parameter capacity with 5M parameter compute cost. |
| **MLA Workspace** | Latent KV Compression | Compresses massive context traces into a fixed-size latent bottleneck $c_t$. |
| **Parallel SSD** | Hardware-Aware Scans | Converts $O(N)$ sequential state loops into $O(\log N)$ parallel GPU kernels. |
| **Hybrid Inject** | O((T/4)² Attention) | Injects exact short-term recall without breaking linear sequence complexity. |
| **DendriticMixer** | K-Branch Gated Mixing | Biologically-motivated feature selection before temporal scans. |

## Parameter Distribution (Default Config)

| Component | Parameters | % Total |
|---|---|---|
| Total | **20,151,355** | 100% |
| Fast Layers (MoE) | 9,485,840 | 47.1% |
| Slow Layers (MoE) | 8,622,088 | 42.8% |
| Attention & MLA | 731,682 | 3.6% |
| Embed & Head | 1,311,744 | 6.5% |

## Quick Start (Free Tier Training)

NEURON-1 is optimized for training on **Free Cloud GPUs** (Google Colab T4 / Kaggle).

```bash
# 1. Setup Environment
pip install torch transformers datasets

# 2. Run Comprehensive Architecture Test
python tests/TEST1.PY --quick

# 3. Start Free-Tier Training (TinyStories Curriculum)
# This script streams data and auto-saves checkpoints to Google Drive
python train.py
```

## Project Structure

```text
neuron1/
  layers.py          — Parallel SSD, SparseMoE, MLA, and Dendritic components
  model.py           — Hybrid assembly with temporal striding
  config.py          — Modernized MoE/MLA configuration flags
  data.py            — Streaming data loaders + vocab
  train.py           — Colab-optimized curriculum training loop
tests/
  TEST1.PY           — Scaling law proxy and associative benchmark suite
```

## Implementation Notes
The architecture uses **Structured State Space Duality (SSD)** for its fast layers. This reformulates the True Delta Rule into a parallel associative scan using scalar decay, enabling massive training speedups on modern hardware while maintaining strong semantic logic and multi-hop reasoning capabilities.

---
*NEURON-1: Bridging the gap between Recurrent Efficiency and Attention Precision.*
