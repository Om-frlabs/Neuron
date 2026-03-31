# SECTION 6 — IMPLEMENTATION ROADMAP

## Week 1: Architecture Scaffold (40 GPU-hours estimated)

### Deliverables
- [ ] Complete PyTorch implementation of all novel components
- [ ] Unit tests for every layer type
- [ ] FLOP/memory profiling validation

### Detailed Plan

**Day 1-2: Core Components**
```
Libraries:
  torch >= 2.1.0
  einops >= 0.7.0
  triton >= 2.1.0 (for custom kernels)
  wandb (logging)
  pytest (testing)

Implement:
  1. CompNorm layer → test: output shape, gradient flow, 
     sparsity at different temperatures
  2. DendriticMixer → test: each dendrite receives correct 
     feature subsets, gradients flow through all branches
  3. DeltaMemory → test: state updates follow delta rule, 
     surprise gating modulates alpha correctly
  4. GatedLRU → test: recurrent dynamics stable over 1000 steps
```

**Day 3-4: Architecture Assembly**
```
Implement:
  5. FastLayer (with predictive residual, temporal stride)
  6. SlowLayer  
  7. GlobalWorkspace bottleneck
  8. Full Neuron1 model
  
Tests:
  - Forward pass produces correct output shape
  - Parameter count matches derivation (±1%)
  - State carry-over between sequences works correctly
  - Gradient norms are reasonable (no explosion/vanishing)
```

**Day 5: Profiling**
```
Profile:
  - FLOPs per token (target: ~3.5M)
  - Memory per token (target: ~20KB state)
  - Inference latency on CPU (target: <100ms/token)
  - Training throughput on single GPU (target: >50K tokens/s on T4)
  
Tools: torch.profiler, custom FLOP counter, time.perf_counter
```

### Failure Modes & Go/No-Go

| Failure | Detection | Mitigation |
|---|---|---|
| Gradient explosion in DeltaMemory | Loss NaN within 100 steps | Clamp state norm; reduce alpha |
| CompNorm collapse (all weight on one neuron) | Monitor activation sparsity | Increase temperature τ; add entropy regularizer |
| Temporal stride causing information loss | Validation loss higher than stride=1 baseline | Reduce stride; switch to average-pooling |
| Parameter count exceeds budget | Count > 5.5M | Reduce d_state or n_dendrites |

**Go/No-Go at end of Week 1:** If single forward pass runs correctly and profiling meets targets, proceed. If DeltaMemory is unstable, fallback to standard GLA for fast layers.

---

## Week 2: Toy Training Validation (80 GPU-hours)

### Deliverables
- [ ] Successful training on TinyStories and Shakespeare
- [ ] Loss curves + comparison with baseline transformer
- [ ] Ablation results for each architectural hypothesis

### Detailed Plan

**Day 1-2: TinyStories Training**
```
Dataset: TinyStories (Eldan & Li, 2023)
  - ~2M short stories, ~500M tokens
  - Perfect for validating narrative coherence at tiny scale

Configuration:
  batch_size: 64
  seq_len: 256
  lr: 1e-3, cosine decay
  warmup: 500 steps
  total: 20K steps (~1.3B tokens with repetition)

Success criteria:
  - Loss < 3.0 by step 5K
  - Generated stories are coherent 3+ sentences
  - Loss competitive with same-size transformer baseline
```

**Day 3: Shakespeare Training**
```
Dataset: TinyShakespeare (~1M tokens)
  - Validates: style capture, vocabulary precision, meter

Configuration:
  Same as above but seq_len: 512
  Total: 10K steps

Success criteria:
  - Generated text captures iambic rhythm
  - Character names used consistently
```

**Day 4-5: Ablation Study**

| Ablation | Change | Measure |
|---|---|---|
| No DeltaMemory (fast layers use standard recurrence) | Replace DeltaMemory with GatedLRU in fast layers | ICL performance, loss |
| No CompNorm (use standard LayerNorm) | Replace CompNorm with LayerNorm | Activation sparsity, loss |
| No Global Workspace (direct connection) | Remove bottleneck, d=256 throughout | Representation compression, loss |
| No Predictive Residual (standard residual) | Standard skip connections | Per-layer mutual information |
| No Temporal Stride (stride=1 everywhere) | All layers stride=1 | Training speed, loss |
| No Dendritic Mixing (standard MLP mixing) | Replace DendriticMixer with Linear | Compositional generalization score |
| **Frozen vs. Trainable Slow Layers** | (a) Fully frozen at init, (b) Frozen after 5K steps, (c) Never frozen | Slow-layer validation loss; downstream generalization |

Each ablation: train for 5K steps on TinyStories, measure validation loss + specific probing metrics.

> **⚠️ Blocker 3 Gate:** The frozen-layers ablation is a **go/no-go requirement**. If freezing slow layers increases validation loss by >10% compared to fully-trainable, abandon the freeze strategy and implement **gradual freeze** (see Section 3, Hypothesis C).

### Failure Modes & Go/No-Go

| Failure | Detection | Pivot |
|---|---|---|
| NEURON-1 underperforms transformer baseline by >10% | Validation loss comparison | Analyze ablations; remove weakest innovations |
| DeltaMemory doesn't improve ICL | Ablation shows no benefit | Replace with GLA + standard attention hybrid |
| Training unstable beyond 10K steps | Loss oscillation or divergence | Reduce learning rate; add gradient clipping; simplify |

**Go/No-Go at end of Week 2:** If NEURON-1 matches or beats the transformer baseline on TinyStories AND at least 3/6 ablations show clear benefit, proceed to full training. Otherwise, simplify architecture based on ablation results.

---

## Week 3: Curriculum Training on Full Dataset (200 GPU-hours)

### Compute Resources
```
Primary: Google Colab Pro (T4 GPU, ~8h sessions)
Backup: Kaggle TPU v3-8 (30h/week, free)
Realistically: 25 Colab sessions × 8h = 200 GPU-hours target
  BUT expect ~40% session failure/preemption overhead
  Actual calendar estimate: 35-40 sessions → 18-20 calendar days
Cost: ~$50 (Colab Pro monthly subscription)

⚠️ Kaggle Note: Kaggle TPUs are free but have non-commercial 
output restrictions. Verify license terms before using 
TPU-generated checkpoints in published work.

PARALLEL COMPUTE STREAMS (Critical scheduling note):
  Stream A: Colab T4 → curriculum training (Phases 1-4)
  Stream B: Local RTX 3050 → Qwen2.5-7B teacher generation (~18h)
  These are INDEPENDENT compute streams running simultaneously.
  Start Stream B on Day 1 of Week 3 so teacher-generated data 
  is ready before Phase 3 begins (Day 5). Do NOT serialize these —
  the 3050 should be generating distillation data while the T4 
  trains on Phase 1-2 data.
```

### Training Schedule

| Day | Phase | Data | Tokens | Checkpoint |
|---|---|---|---|---|
| Day 1-2 | Phase 1: Foundations | Logic + code + definitions | 1B | ckpt-1B |
| Day 3-4 | Phase 2: Compression | Wikipedia + textbooks + abstracts | 2B | ckpt-3B |
| Day 5-7 | Phase 3: Reasoning | CoT + proofs + code | 3B | ckpt-6B |
| Day 8-9 | Phase 4: Generalization | Multilingual + analogies + diverse | 2B | ckpt-8B |
| Day 10 | Phase 5: Distillation | Teacher-generated traces | 2B | ckpt-10B |

### Monitoring
```
Track every 500 steps:
  - Training loss (per-domain)
  - Validation loss (held-out set from each domain)
  - Activation sparsity (CompNorm temperature)
  - DeltaMemory state norms
  - Global workspace utilization (effective dimensionality)
  - 4 probe tasks: analogy, logic, code, reasoning
```

### Data Preparation
```
Sources (all open-access):
  - Logic: FOLIO, PrOntoQA, LogiQA
  - Code: The Stack (filtered Python/Haskell), CodeContests
  - Math: proof-pile, GSM8K-derivations, MATH
  - Textbooks: OpenTextbook subset, Phi-1 textbook format
  - Wikipedia: Abstract-only dump (~200M tokens)
  - CoT: OpenAssistant filtered for reasoning chains
  - Distillation: Generate locally via Qwen2.5-7B on RTX 3050 ($0)
  - High-quality traces: 10K from GPT-4o-mini API (~$50)
```

---

## Week 4: Distillation + Evaluation (60 GPU-hours)

### Day 1-2: Knowledge Distillation
```
Phase A: Soft-target distillation (500M tokens)
  - Teacher: Qwen2.5-7B running locally on RTX 3050 (4-bit GPTQ)
  - Cost: $0 (local compute, ~6h generation time)
  - Cache teacher logits to disk for reuse
  
Phase B: CoT compression (1B tokens)
  - Primary: Qwen2.5-7B local traces (~12h generation)
  - Premium: 10K hardest problems via GPT-4o-mini API (~$50)
  
Total distillation cost: ~$50 (GPT-4o-mini for 10K premium traces)
```

### Day 3-4: TURING-NANO Benchmark Evaluation
```
For each of 12 TURING-NANO tasks + 4 standard benchmarks (BBH, ARC-AGI, HellaSwag, MMLU-subset):
  1. Generate test set (100 examples each for TURING-NANO)
  2. Run NEURON-1 inference
  3. Run GPT-4o, Claude 3.5 via API (for comparison)
  4. Score using task-specific metrics
  5. Compute both raw scores AND IDS scores
  
Estimate: ~$50 in API costs for frontier model comparison
```

### Day 5: Results Analysis + Paper Outline
```
Note: Full paper writing is a MONTH 2 deliverable, not a single day.
Day 5 deliverables:
  - Complete results tables with confidence intervals
  - Key figures: loss curves, ablation comparisons, IDS charts
  - Paper outline + abstract draft
  - Detailed analysis of failure modes and surprising results
  
Full paper writing timeline: 
  Month 2, Weeks 1-2: Draft (12 pages, NeurIPS/ICML format)
  Month 2, Week 3: Internal review + revision
  Month 2, Week 4: Submission preparation
```

### Total Compute Budget Summary

| Phase | GPU-hours | Cost |
|---|---|---|
| Week 1: Scaffold | 40 | $20 (Colab) |
| Week 2: Toy Training + Ablations | 80 | $10 (Colab) |
| Week 3: Full Training (with 2× buffer) | 200 (plan) / 350 (realistic) | $50 (Colab Pro) |
| Week 4: Distillation + Eval | 60 | $30 (Colab) |
| Local RTX 3050 (teacher gen) | ~20 | $0 (owned hardware) |
| API costs (10K premium traces + eval) | — | $100 |
| **Total** | **~400–550 GPU-hours** | **~$210** |
