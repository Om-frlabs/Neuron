# SECTION 1 — LITERATURE SYNTHESIS MATRIX

## 1.1 Sequence Modeling Paradigms

| Architecture | Core Primitive | Cost | Memory Mechanism | Inductive Bias | Parallelizable | Failure Mode |
|---|---|---|---|---|---|---|
| **Transformer** (Vaswani 2017) | Scaled dot-product attention over all pairs | O(n²d) time, O(n²) memory | No explicit memory; context window IS memory | None (universal approx.) | Fully parallel | Quadratic cost; no length generalization; attention dilution at scale |
| **Linear Attention** (Katharopoulos 2020) | φ(Q)φ(K)ᵀV kernel trick; reformulates attention as RNN | O(nd²) | Recurrent state S∈ℝ^(d×d) | Linear associative memory | Causal: sequential; non-causal: parallel | Catastrophic retrieval failure; cannot do sharp lookup — kernel approximation blurs attention |
| **Performer** (Choromanski 2021) | FAVOR+ random feature maps to approximate softmax kernel | O(nd·r) where r = #features | Random projection state | Positive orthogonal random features | Fully parallel | Approximation degrades for sharp attention patterns; high variance in low-feature regime |
| **Longformer / BigBird** | Sparse attention: sliding window + global tokens + random | O(n·w) where w = window | Sliding window = local memory; global tokens = broadcast | Locality + selective global access | Parallel (sparse ops) | Misses critical long-range dependencies outside window; global tokens are a bottleneck |
| **Mamba / S6** (Gu & Dao 2023) | Selective State Space Model: input-dependent A, B, C matrices | O(n·d·N) where N = state dim | Continuous hidden state h∈ℝ^(d×N) | Sequential dynamics; input-dependent gating | Parallel via scan | State compression bottleneck; struggles with exact retrieval from long contexts; fixed state size |
| **RWKV** (Peng 2023) | Linear recurrence with token-shift mixing; WKV attention analog | O(nd) | Recurrent state with exponential decay | Time-mixing via learned decay + channel mixing | Parallel via chunked computation | Limited associative memory capacity; exponential decay forgets selectively poorly |
| **RetNet** (Sun 2023) | Retention: multi-scale exponential decay attention with γ-decay | O(nd) recurrent; O(n²d) parallel | Multi-scale recurrent states | Multi-timescale processing | Dual form: parallel train, recurrent infer | Decay heads can miss patterns requiring non-decaying memory; less flexible than full attention |
| **xLSTM** (Beck 2024) | Exponential gating + matrix memory (mLSTM) / scalar memory (sLSTM) | O(nd²) for mLSTM | Matrix memory M∈ℝ^(d×d) with covariance update | LSTM-family: sequential gating, write/erase semantics | mLSTM: fully parallel; sLSTM: sequential | Matrix memory limited by d²; exponential gating needs careful stabilization |
| **GLA** (Yang 2024) | Gated Linear Attention: data-dependent decay in linear attention | O(nd·chunk) | Recurrent state with gated updates | Hardware-aware chunked computation | Chunk-parallel | Inherits linear attention's retrieval limitations; chunk boundary artifacts |
| **TTT** (Sun 2024) | Hidden state IS a model; updated via self-supervised gradient steps at inference | O(n·inner_steps·d) | Hidden state = neural network weights | Test-time adaptation; self-supervised learning | Mini-batch parallel within chunks | Expensive inner loop; gradient instability at test time; requires careful learning rate scheduling |
| **Hyena / H3** (Poli 2023) | Long convolutions (implicitly parameterized) + multiplicative gating | O(n log n) via FFT | Implicit in convolution filter | Long-range dependence via learned filters | Fully parallel | Convolution filters hard to learn for sharp positional patterns; gating adds parameters |
| **Griffin** (De 2024) | Real-Gated Linear Recurrent Unit (RG-LRU) + local sliding-window MQA | O(nd) recurrent + O(n·w) local attn | Gated recurrent state + local attention window | Locality + recurrent dynamics | RG-LRU: scan-parallel; local attn: parallel | Local attention window limits global reasoning; RG-LRU has bounded state capacity |
| **Liquid Neural Networks** (Hasani 2022) | ODE-based continuous-time dynamics; neural ODE with input modulation | O(n·solver_steps·d) | Continuous hidden trajectory | Continuous dynamics; causal temporal processing | Limited (sequential ODE solve) | ODE solver overhead; stiff dynamics → instability; hard to parallelize |
| **MEGA** (Ma 2023) | Exponential Moving Average (EMA) + single-head gated attention | O(n·d) EMA + O(n²) gated attn | EMA state + attention context | Multi-scale temporal smoothing via EMA | EMA: parallel; attention: parallel | EMA is a linear smoother — limited nonlinear interaction; single-head attention limits capacity |

### Key Synthesis Insights

1. **The Retrieval-Compression Tradeoff**: Full attention excels at exact retrieval but scales quadratically. All sub-quadratic methods compress history into fixed-size states, trading retrieval precision for efficiency. For a 5M-parameter model, we MUST use sub-quadratic methods but need to maximize retrieval fidelity within the state.

2. **Input-Dependent Gating is Universal**: Mamba, xLSTM, GLA, Griffin, RWKV all converge on input-dependent gating as the key mechanism. This is the modern analog of attention — routing information conditionally — but in O(n) time.

3. **The Duality Principle** (Mamba-2): SSMs and linear attention are mathematically dual. Any linear recurrence can be viewed as structured linear attention with a specific mask. This means architecture design should focus on the *structure of the mask/decay*, not the formalism.

4. **Fast Weights as In-Context Learning**: DeltaNet and TTT show that writing to model weights at inference time enables powerful in-context learning. The delta rule (error-correcting Hebbian update) is more surgical than additive linear attention.

---

## 1.2 Brain-Inspired & Biological Principles

### Predictive Coding (Rao & Ballard 1999; Friston 2010)

**Mathematical formulation:** Hierarchical generative model where layer l predicts the activity of layer l-1:
- Prediction: μ_l = f_θ(r_l) where r_l is the representation at layer l
- Error: ε_{l-1} = r_{l-1} − μ_l  
- Update: r_l ← r_l + α(∂/∂r_l)[−‖ε_{l-1}‖² − ‖ε_l‖²]

**Algorithmic essence:** Top-down predictions + bottom-up errors. Only error signals propagate, not raw data. This compresses communication bandwidth between layers by orders of magnitude. The brain processes ~10⁷ bits/s of sensory input but only ~50 bits/s reach conscious processing — predictive coding explains this compression.

**Relevance to NEURON-1:** In a parameter-starved model, predictive coding could dramatically reduce the information that must flow between layers, forcing compressed representations.

### Hierarchical Temporal Memory (Hawkins & Ahmad)

- **Sparse Distributed Representations (SDRs):** Vectors with ~2% active bits. Overlap = semantic similarity. Extremely noise-robust and memory-efficient.
- **Temporal pooling:** Cells within a column represent different temporal contexts of the same input → the model naturally learns sequences.
- **Union pooling:** Representing multiple predictions simultaneously via the union of SDRs.

**Algorithmic essence:** Sparse, high-dimensional binary codes with temporal context built into the representation structure itself.

### Neocortical Column Structure

- **L2/3:** Lateral connections, feedback from higher areas → contextual modulation
- **L4:** Primary thalamic input → initial feature extraction
- **L5:** Main output to subcortical structures → action/prediction
- **L6:** Feedback to thalamus → timing, attentional gating

**For NEURON-1:** This suggests a layer-specialized architecture where different layers in the stack serve specific computational roles, not generic feed-forward transformation.

### Dendritic Computation

**Mathematical formulation:** Each neuron has K dendritic segments. Segment k integrates a sparse subset S_k of inputs:
- d_k = σ(Σ_{i∈S_k} w_{ki} · x_i)  
- Output: y = g(Σ_k d_k) where g is a nonlinear activation

Each dendrite acts as a nonlinear coincidence detector — detecting specific patterns independently before integration at the soma. This gives each neuron K separate "attention heads" with O(K·|S|) cost instead of O(n²).

### Lateral Inhibition + Winner-Take-All

**Mathematical formulation:**
- y_i = ReLU(x_i − β · max_{j≠i}(x_j))
- Or softer: y_i = x_i · exp(x_i/τ) / Σ_j exp(x_j/τ) (softmax competition)

Enforces sparsity at every layer. Only top-k activations survive. This is computationally equivalent to top-k sparsification but biologically motivated and differentiable.

### Global Workspace Theory (Baars, Dehaene)

**Algorithmic essence:** A narrow information bottleneck through which all modules must broadcast. Only one "coalition" of modules can access the workspace at a time. This forces:
1. Extreme compression of representations
2. Competition between modules for workspace access
3. Global coherence through broadcast

**For NEURON-1:** A narrow bottleneck layer (d=64) between encoder/decoder halves forces the model to learn maximally compressed, compositional codes.

### Grid Cells & Place Cells

**Mathematical formulation:** Grid cells exhibit periodic firing patterns that tile space with hexagonal grids at multiple scales. Computationally, these implement:
- Modular arithmetic on continuous variables
- Multi-resolution spatial encoding 
- Translational symmetry in latent space

**For NEURON-1:** Geometric regularization — encouraging the hidden state to maintain structured, periodic geometry — can improve compositional generalization by providing a "coordinate system" for concept space.

### Memory Consolidation (Hippocampus → Neocortex)

Fast (hippocampal) learning rapidly encodes episodic memories. Slow (neocortical) learning gradually extracts statistical regularities through replay. **Two-timescale memory** is the biological solution to the stability-plasticity dilemma.

---

## 1.3 Efficiency & Compression Principles

| Principle | Core Insight | Implication for NEURON-1 |
|---|---|---|
| **Lottery Ticket Hypothesis** (Frankle & Carlin 2019) | Dense networks contain sparse subnetworks that, when trained in isolation, match full performance. | Every parameter in NEURON-1 must be "load-bearing." Enforce structural sparsity from initialization. |
| **Information Bottleneck** (Tishby) | Optimal representations maximize I(Z;Y) while minimizing I(Z;X) — compress input, preserve task-relevant info. | The global workspace bottleneck directly implements this: force Z to be low-dimensional. |
| **Kolmogorov Complexity / MDL** | The best model is the shortest program that generates the data. | At 5M params, the model IS a short program. Every parameter must encode a reusable rule, not a memorized fact. |
| **Chinchilla Scaling** (Hoffmann 2022) | Optimal: ~20 tokens per parameter. For 5M params → 100M tokens optimal. | We budget 5-10B tokens (1000x over-training) to force maximum compression of the data into the parameters. |
| **Knowledge Distillation** (Hinton 2015) | Soft targets from a teacher carry "dark knowledge" about inter-class relationships. | Distilling from GPT-4/Claude can transfer reasoning patterns, not just answers. |
| **Flash Attention** (Dao 2022) | Hardware-aware algorithms can dramatically reduce wall-clock time even when FLOP count is the same. | All operations must be designed for hardware efficiency — memory-bound vs compute-bound awareness. |
| **BitNet / 1-bit** | Extreme quantization (±1 weights) with full-precision activations can work. | Post-training quantization to 4-bit or 2-bit could make NEURON-1 run on microcontrollers. |

---

## 1.4 Emerging Paradigms (2024–2025)

### Titans (Google, 2025)
**Core innovation:** Neural Long-Term Memory module — a deep neural network that serves as memory, updated at test time using a **surprise-based** write signal. When the model encounters data far from its current memory state (high prediction error / surprise), it writes to memory more strongly. Includes momentum-based surprise for capturing context around surprising events.

**Key insight for NEURON-1:** Surprise-gated memory writes are a Hebbian-compatible mechanism that can be implemented without backpropagation at inference time. The "surprise" metric (prediction error magnitude) is computationally cheap and biologically plausible.

### Mamba-2 / State Space Duality
**Core innovation:** Proves that selective SSMs are equivalent to a structured form of linear attention with a specific semi-separable matrix mask. This duality enables:
- Parallel training via matrix formulation
- Efficient inference via recurrent formulation
- Larger state dimensions via tensor-core-friendly computation

**Key insight for NEURON-1:** We don't need to choose between SSM and attention — they are the same computation viewed differently. Design the *structure* (mask/decay pattern), then choose the formulation based on hardware.

### DeltaNet / Gated DeltaNet
**Core innovation:** Replaces additive linear attention updates with the delta rule: ΔW = α(v - Wq)qᵀ. This performs error-corrective updates to the fast-weight memory, enabling precise overwriting of previously stored associations.

**Key insight for NEURON-1:** The delta rule is the best known mechanism for in-context associative memory in linear-time models. It directly implements error-correcting Hebbian learning.

### HGRN2 / Hawk / Eagle / Jamba / Zamba2
These represent the emerging consensus: **hybrid architectures** combining efficient recurrence (for speed) with selective attention (for precision). The debate is no longer "attention vs. recurrence" but "what ratio and how to combine."

- **Jamba** (AI21, 2024): Interleaves Mamba SSM layers with Transformer attention layers + MoE. Demonstrates that even sparse attention every N-th layer suffices for quality retention.
- **Zamba2** (Zyphra, 2024): Shared attention layer concept — a single attention layer is reused across the depth stack, with unique Mamba layers between. Proves that attention parameters can be amortized across depth, freeing parameter budget for recurrent processing.

### Mixture of Depths (MoD)
**Core innovation (Raposo et al., 2024):** Not all tokens need the same computational depth. MoD routes each token through a *variable number of layers* using a learned binary routing decision per token per layer. Tokens that are "easy" (predictable) skip layers entirely; "hard" tokens get full depth.

**Key insight for NEURON-1:** MoD is a **parameter-free** way to implement adaptive compute — it adds negligible parameters (just a scalar router per layer) but can reduce effective FLOPs by 30-50% on average text. For a tiny model, this is extremely attractive: it lets NEURON-1 concentrate its limited capacity on the tokens that matter most. **Consider adding MoD routing to slow layers in NEURON-1.5.**

### Emergent Capabilities Research
Recent work (Schaeffer et al. 2024) suggests that "emergence" is often a metric artifact — capabilities develop smoothly but appear sudden with nonlinear metrics. For tiny models, this means:
- Don't expect sharp phase transitions
- Focus on *continuous improvement* of reasoning with scale
- Compositional generalization may require specific architectural support, not just scale
