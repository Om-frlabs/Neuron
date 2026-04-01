# 🧠 NEURON-1 Architecture

**Created by:** Om Mishra (@Om-frlabs)

NEURON is a bespoke, causally-strict, **Sparse Mixture of Experts (MoE)** foundation model built entirely from scratch in PyTorch. It is designed to be a highly efficient, production-ready neural network capable of extreme hardware scaling—from $0 Google Colab environments all the way up to multi-core Google TPU (TRC) supercomputers.

---

## 🏗️ Core Architectural Innovations

Unlike standard HuggingFace wrappers, NEURON is engineered from the ground up to solve memory bottlenecks and context scaling using proprietary layer designs.

### 1. `FastLayer` (Temporal Strided Convolutions)
To bypass the quadratic memory bottleneck of traditional Attention mechanisms, NEURON implements a massive hardware optimization called the **FastLayer**. By utilizing 1D Strided Temporal Convolutions (`kernel_size=3`, `stride=2`), the network dynamically compresses long sequences into hierarchical states. This allows the model to "remember" extreme mathematical logic without running out of standard GPU VRAM.

### 2. Sparse Mixture of Experts (MoE)
NEURON utilizes a custom MoE routing engine. The config scales up to **32 independent parameter experts**, but dynamically activates only **the top 2 experts per token**. 
- **The Result:** The model mathematically holds 1.5+ Billion parameters of knowledge, but runs inference at the speed and energy cost of a 100-Million parameter model.

### 3. Causal Hybrid Attention (`SlowLayer`)
Coupled with the `FastLayer`, NEURON implements a strict causal standard Multi-Head Attention layer. Complex gradient bugs (where the model "cheated" by looking ahead at padded sequences) were successfully eliminated by enforcing a dynamic lower-triangular causal mask `tril` before the softmax is computed.

---

## 📈 The 3-Phase Curriculum Training Pipeline

To forge an intelligent model on constrained hardware (trained exclusively on 1.5 hours/day Colab limits), NEURON utilizes a meticulously staged curriculum:

### Phase 1: Syntactic Foundation (`train.py`)
- **Dataset:** `roneneldan/TinyStories`
- **Goal:** Teaching the bare architecture strict English grammar, syntax, and narrative continuity.

### Phase 2: Knowledge Expansion (`train_phase2.py`)
- **Dataset:** `wikipedia` / Math / Code
- **Goal:** The model absorbs millions of tokens of hard world-knowledge and logical theorems. 
- **Compound Loss Function:** Implements a heavily optimized `Neuron1Loss` that penalizes Cross-Entropy, while simultaneously enforcing an `expert_load_balancing_loss` to ensure all 32 MoE experts share the parameter load equally.

### Phase 3: Supervised Fine-Tuning (SFT) (`train_phase3.py`)
- **Dataset:** `OpenAssistant/oasst_top1_2023-08-25` + Synthetic Identity Injection
- **Goal:** The raw predictive model is converted into an interactive Chat Assistant. 
- The training loop enforces strict `<|im_start|>` and `<|im_end|>` ChatML tokens. During inference (`chat.py`), the generation loop dynamically catches these tags to prevent runaway self-dialogue hallucinations.

*(See the `NEURON_Checkpoints_SFT` directory for the resulting instruction-tuned weights.)*

---

## 🚀 Extreme Scaling (Google TPU Integration)

The `train_tpu.py` harness is included specifically for distributed supercomputing on the **Google TPU Research Cloud (TRC)**.

By utilizing `torch_xla` and `xmp.spawn`, NEURON transitions from single-GPU Colab to an **8-core TPU multiprocessing environment**. 
- The `MpDeviceLoader` allows the dataset to bypass CPU bottlenecks and stream directly into TPU High-Bandwidth Memory (HBM).
- The `xm.optimizer_step()` forces 8 separate PyTorch processes to synchronize their gradients across the massive TPU interconnect bridge in milliseconds.

**To run the 1.5 Billion Parameter scale-up on a Google TPU cluster:**
```bash
python train_tpu.py
```

---
*Built by a single engineer proving that frontier AI architecture can be engineered outside of silicon valley clusters.*
