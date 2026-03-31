# SECTION 2 — INTELLIGENCE DEFINITION & TURING-NANO BENCHMARK

## 2.1 What IS Intelligence at 4–7M Parameters?

At this parameter scale, the model cannot memorize — it must **compress**. Intelligence at 5M parameters is the model's ability to extract and reuse abstract rules from data, measured by how efficiently each parameter contributes to generalizable computation.

### Formal Definition

**Intelligence Density (ID)** of a model M with |θ| parameters:

```
ID(M) = (1/|θ|) · Σ_t  w_t · Score_t(M)
```

where Score_t is the normalized score on task t and w_t reflects task difficulty.

### Six Axes of Small-Scale Intelligence

| Axis | Definition | Measurement | Why It Matters at 5M |
|---|---|---|---|
| **(a) Information-Theoretic Capacity** | Bits of useful mutual information I(θ; D) compressed into parameters | Probe accuracy vs. parameter count on held-out data | Every bit must be useful; no room for redundant circuits |
| **(b) Compositional Generalization** | Recombining learned primitives to solve unseen combinations | SCAN/COGS-style accuracy; ARC-AGI pattern completion | Tests whether model learned RULES, not patterns |
| **(c) In-Context Learning (ICL) Efficiency** | Shots-to-task-acquisition: how many examples to learn a new task in-context | k-shot accuracy curve slope on novel tasks | Small models need maximally efficient ICL — every example counts |
| **(d) Reasoning Depth** | Quality of multi-step inference chains | Chain-of-thought accuracy on multi-hop problems | Tests whether parameters encode reasoning procedures |
| **(e) Calibration** | P(correct \| confidence=p) ≈ p; knowing what it doesn't know | Expected Calibration Error (ECE) | Tiny models SHOULD be uncertain about many things — honesty is intelligence |
| **(f) Transfer** | Zero-shot OOD performance without fine-tuning | Cross-domain zero-shot evaluation | Tests whether compressed knowledge generalizes |

---

## 2.2 TURING-NANO Benchmark Suite

### Design Principles
- All tasks are **parameter-budget-aware**: scoring normalizes by |θ|
- Tasks test **reasoning**, not **recall** — no factual knowledge questions
- Each task has a **ceiling** achievable by a perfect reasoner regardless of parameter count
- Scoring: **Intelligence Density Score (IDS)** = raw_score / log₂(|θ|)

### The 12 Tasks

| # | Task | Category | Description | Scoring | Ceiling |
|---|---|---|---|---|---|
| 1 | **LogicGrid** | Logical Reasoning | Solve 3×3 logic grid puzzles from natural language clues | Exact match accuracy | 100% |
| 2 | **AnalogyMap** | Analogy | A:B::C:? with abstract relationships (not word2vec-solvable) | Top-1 accuracy | 100% |
| 3 | **SeqComplete** | Pattern Completion | Continue integer/symbol sequences requiring rule induction | Exact match on next 3 elements | 100% |
| 4 | **CausalChain** | Causal Inference | Given a causal graph description, predict intervention outcomes | Accuracy on do-calculus queries | 100% |
| 5 | **FewShotConcept** | Few-Shot Learning | Learn a novel binary concept from 4 examples, classify 10 test items | Classification accuracy | 100% |
| 6 | **TinyCode** | Code Completion | Complete 5-line Python functions given docstring + first 2 lines | Pass@1 on test cases | 100% |
| 7 | **MathInduct** | Mathematical Induction | Given base case + rule, compute f(n) for n=5 | Exact numerical match | 100% |
| 8 | **ComprehendMini** | Language Understanding | Reading comprehension on 50-word passages, 3 questions each | F1 score | 100% |
| 9 | **SpatialNav** | Spatial Reasoning | Navigate a described 2D grid ("go north 3, east 2, where am I?") | Distance from correct position (inverted) | 100% |
| 10 | **CommonSenseQ** | Commonsense | Physical/social commonsense: "Can you carry a car?" — yes/no with explanation | Accuracy + explanation quality (**rubric-based human scoring**, NOT LLM-judge — since NEURON-1 is distilled from frontier models, using them as judges creates circular evaluation bias) | 100% |
| 11 | **FactCompress** | Factual Recall Efficiency | Given 20 facts during context, quiz on 5 random facts after | Recall accuracy (measuring bits/param of episodic memory) | 100% |
| 12 | **MetaAdapt** | Meta-Learning | Given a distribution shift mid-context, adapt predictions | Post-shift accuracy improvement rate | 100% |

### Scoring Protocol

```
For each model M on task t:
  raw_score_t = task-specific metric (0 to 100)
  
Intelligence Density Score:
  IDS_t(M) = raw_score_t / log₂(|θ_M|)

Overall TURING-NANO Score:
  TN(M) = (1/12) · Σ_{t=1}^{12} IDS_t(M)

Comparative Analysis:
  For GPT-4o  (|θ| ≈ 1.8T):  log₂(1.8T) ≈ 40.7
  For Claude 3.5 (|θ| ≈ 200B): log₂(200B) ≈ 37.5
  For NEURON-1 (|θ| ≈ 5M):   log₂(5M)   ≈ 22.3
  
  → NEURON-1 gets a 1.83x scoring advantage over GPT-4o
  → To beat GPT-4o on IDS, NEURON-1 needs raw_score > GPT-4 raw / 1.83
     i.e., ~55% of GPT-4's raw score suffices for IDS parity
```

### Why This Benchmark Design Works

1. **Log-scaling normalization** reflects diminishing returns of parameters — each doubling contributes less, so small models get fair credit
2. **No factual recall tasks** (except FactCompress, which tests in-context episodic binding, not pre-trained knowledge)
3. **Compositional and reasoning tasks** are exactly where structured inductive biases should outperform brute-force memorization
4. **Meta-learning task** tests the unique capability of models with fast-weight / TTT-style adaptation

### Mandatory Standard Benchmark Comparison

> **⚠️ Peer Review Requirement:** TURING-NANO alone is insufficient for publication — reviewers will flag that the normalization scheme was designed by the same team proposing NEURON-1. To establish credibility, NEURON-1 MUST also be evaluated on:

| Benchmark | What It Tests | Why It's Needed |
|---|---|---|
| **BIG-Bench Hard (BBH)** | 23 challenging reasoning tasks from BIG-Bench | Industry-standard reasoning benchmark; allows direct comparison with published results from Phi, Gemma, etc. |
| **ARC-AGI** | Abstract pattern completion grids | Gold standard for compositional generalization; directly tests NEURON-1's claimed strength |
| **HellaSwag** | Sentence completion / commonsense | Standard LM quality benchmark; normalizes against published model cards |
| **MMLU (5-shot, subset)** | Multi-task language understanding | Expected by reviewers; even poor absolute scores show the per-parameter efficiency claim honestly |

Report both **raw scores** and **IDS (score/log₂|θ|)** on all benchmarks. The honest expectation: NEURON-1 will score low in absolute terms on MMLU/HellaSwag but should be competitive on ARC-AGI and BBH per-parameter.
