# SECTION 7 — OPEN QUESTIONS & FUTURE RESEARCH

## 7.1 What Does NEURON-1's Success or Failure Tell Us?

### If NEURON-1 Succeeds (IDS > GPT-4):

**Implication 1: Intelligence is algorithmic, not statistical.**
Transformers learn intelligence as an emergent byproduct of next-token prediction at scale. If NEURON-1 achieves higher per-parameter intelligence, it means that encoding the right computational primitives (dendritic mixing, predictive coding, dual-timescale memory) directly is more efficient than hoping they emerge. Intelligence is not a curve to be scaled — it is a program to be written.

**Implication 2: The brain's architecture IS the intelligence.**
Neuroscience's structural principles (cortical columns, dendritic computation, hippocampal-neocortical complementarity) are not just biological constraints — they are computational optimizations honed by 500 million years of evolution. If biologically-inspired inductive biases outperform vanilla architectures at small scale, it suggests these structures are close to optimal for natural intelligence.

**Implication 3: Data efficiency is a solvable problem.**
The fact that humans learn language from ~10M tokens while GPT-4 needs 13T suggests a >1,000,000× efficiency gap. If NEURON-1 closes even 10% of this gap, it implies that curriculum design and architectural priors can substitute for data volume — transformative for low-resource settings.

### If NEURON-1 Fails (IDS < GPT-4):

**Implication 1: Scale IS intelligence (the bitter lesson).**
This would support Sutton's Bitter Lesson: general methods that leverage computation dominate engineered solutions. The biological priors might be wrong priors for silicon-based computation.

**Implication 2: Minimum parameter thresholds exist.**
There may be irreducible parameter requirements for certain reasoning capabilities — phase transitions that 5M parameters cannot cross regardless of architecture. This would constrain the "small model" research agenda.

**Implication 3: The fast-weight mechanism needs hardware co-design.**
The delta-rule fast-weight system, while theoretically elegant, may be impractical without custom hardware support (dedicated fast-weight memory buses). Pure software implementation may be too slow.

---

## 7.2 Open Research Questions

### Architecture Questions

1. **Optimal dendrite count as a function of model width:**
   Is K=4 optimal for d=256, or is there a scaling law K(d) that maximizes compositional capacity per parameter?

2. **Predictive coding depth:**
   Should predictive residuals span more than one layer? Multi-scale predictive coding (layer l predicts layer l-2, l-3) might improve but at what parameter cost?

3. **Global workspace dimensionality:**
   Is d_gw=64 optimal? Information-theoretically, the workspace should match the "effective dimensionality of thought." Is there an empirical way to discover this?

4. **Temporal hierarchy design:**
   Fixed strides [1,1,2,2,4,4,1,1] vs. learned strides? Could the model learn its own temporal hierarchy through differentiable downsampling?

### Training Questions

5. **Curriculum ordering sensitivity:**
   How sensitive is final performance to the exact ordering of curriculum phases? Is "foundations → compression → reasoning → generalization" provably optimal, or would "reasoning → foundations → generalization → compression" work equally well?

6. **Distillation saturation point:**
   At what point does distillation from a larger teacher provide diminishing returns for a tiny student? We hypothesize ~2B tokens, but this needs empirical validation.

7. **Self-distillation loops:**
   Could NEURON-1 distill from its own best checkpoints iteratively? Previous-best → current → next-best, creating an autonomous improvement loop?

### Theoretical Questions

8. **Is there a minimum parameter count for compositional generalization?**
   PAC-learning theory suggests bounds, but these are loose. Can we tighten the bound for specific task families (like ARC-AGI)?

9. **Fast-weight convergence guarantees:**
   Does the surprise-gated delta rule converge to the correct associative memory under the streaming (online) setting? What are the regret bounds compared to optimal in-hindsight retrieval?

10. **Information bottleneck tightness:**
    Does the global workspace provably achieve the information bottleneck optimum for a given bottleneck width? Or does it get stuck in local optima?

---

## 7.3 Extensions and Future Work

### Near-term (Months 2-3)

| Extension | Description | Expected Impact |
|---|---|---|
| **NEURON-1.5** | Increase to 10M params; add local attention in slow layers | +15% on TURING-NANO |
| **NEURON-1-Code** | Domain-specific variant for code synthesis | Competitive with Phi-1 on code tasks |
| **NEURON-1-Math** | Mathematics specialist with proof-step tokenizer | Better GSM8K per-parameter score |
| **Hardware optimization** | Custom Triton kernels for dendritic mixing + delta memory | 3× inference speedup |

### Medium-term (Months 4-6)

| Extension | Description |
|---|---|
| **NEURON-2** | Scale to 50M params; introduce mixture-of-dendrites (conditional activation of dendrite subsets) |
| **Multi-modal NEURON** | Vision encoder → workspace → language decoder; test on VQA tasks |
| **Online learning NEURON** | Deploy with continuous fast-weight adaptation; measure knowledge accumulation over time |

### Long-term (Year 2+)

| Extension | Description |
|---|---|
| **NEURON Hardware** | Custom ASIC with native fast-weight memory and dendritic compute units |
| **NEURON Swarm** | Multiple NEURON-1 instances communicating through shared workspaces — emergent collective intelligence |
| **Biological validation** | Compare NEURON-1's learned representations with neural recordings; validate biological fidelity |

---

## 7.4 Philosophical Implications

If a 5M parameter model can achieve non-trivial reasoning, this constrains theories of consciousness and intelligence:

1. **Against the "intelligence requires scale" hypothesis:** Fruit flies (100K neurons) exhibit learning, memory, navigation, and social behavior. C. elegans (302 neurons) demonstrates associative learning. Intelligence clearly does not require massive scale in biology — why should it in silicon?

2. **The compression-consciousness connection:** Global workspace theory suggests that consciousness arises from information compression and broadcast. NEURON-1's workspace bottleneck implements exactly this — not claiming consciousness, but noting that the computational structure is isomorphic.

3. **The minimum description length of thought:** If NEURON-1 can reason in 5M parameters (20MB), this puts an upper bound on the Kolmogorov complexity of the reasoning algorithm itself. Intelligence-as-program may be surprisingly short.

---

## 7.5 Key Citations

| Paper | Key Insight for NEURON-1 |
|---|---|
| Vaswani et al. (2017) "Attention Is All You Need" | Baseline: global attention, O(n²), no memory |
| Gu & Dao (2023) "Mamba" | Input-dependent SSM; selective state spaces |
| Dao et al. (2024) "Mamba-2" | State space duality; SSMs = structured attention |
| Yang et al. (2024) "Gated DeltaNet" | Delta rule for precise associative memory |
| Beck et al. (2024) "xLSTM" | Exponential gating + matrix memory |
| Sun et al. (2024) "TTT Layers" | Hidden state as learnable model |
| De et al. (2024) "Griffin" | RG-LRU + local attention hybrid |
| Google (2025) "Titans" | Surprise-based neural long-term memory |
| Rao & Ballard (1999) | Predictive coding: error-driven processing |
| Friston (2010) | Free energy principle; surprise minimization |
| Dehaene et al. (2011) | Global workspace theory of consciousness |
| Tishby et al. (2000) | Information bottleneck principle |
| Frankle & Carlin (2019) | Lottery ticket hypothesis |
| Hoffmann et al. (2022) | Chinchilla scaling laws |
| Eldan & Li (2023) | TinyStories: coherent tiny LMs |
| Li et al. (2024) | "Textbooks Are All You Need" (Phi-1) |
| Deletang et al. (2024) | "Language Modeling Is Compression" |
| Olshausen & Field (1996) | Sparse coding in visual cortex |
| Hawkins & Ahmad (2016) | HTM: SDRs + temporal memory |
| Poli et al. (2023) | Hyena: long convolutions + gating |
| Ma et al. (2023) | MEGA: EMA + gated attention |
| Peng et al. (2023) | RWKV: linear attention RNN |
| Sun et al. (2023) | RetNet: retention mechanism |
| Hasani et al. (2022) | Liquid networks: ODE dynamics |
| Hinton et al. (2015) | Knowledge distillation |
| Rissanen (1978) | Minimum Description Length |
