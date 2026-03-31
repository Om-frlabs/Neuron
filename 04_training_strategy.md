# SECTION 4 — TRAINING STRATEGY FOR TINY INTELLIGENCE

## STAGE 0 — TOKENIZER DESIGN

### Recommendation: Custom Byte-Level BPE, Vocab Size = 4,096

**Justification:**

Standard BPE tokenizers (GPT: 50K, Llama: 32K) waste catastrophic parameter budget on embedding tables:
- 50K vocab × 256 dim = **12.8M parameters** — more than NEURON-1's entire budget
- 32K vocab × 256 dim = **8.2M parameters** — still too large

**Optimal vocab size analysis:**

```
Embedding cost = V × d_model
Useful capacity = f(V) where f is the information gained per token

For d_model = 256:
  V = 4096:  embedding = 1.05M params (20% of 5.24M budget)
  V = 8192:  embedding = 2.10M params (40% — too expensive)
  V = 2048:  embedding = 0.52M params (10% — but tokens too short, 
             sequences become long, hurting recurrent memory)
  V = 4096:  OPTIMAL BALANCE
```

**Tokenizer specifics:**
- **Byte-level BPE** with 4096 merges trained on the target curriculum data
- **Morpheme-aware merges**: bias merge decisions toward morpheme boundaries where possible (using a morphological dictionary as prior)
- **Special tokens**: `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`, `<THINK>` (for chain-of-thought), `<SEP>` — 6 total
- **Information density**: Targeting 4.5 bits/token (vs. 3.8 for GPT-4's tokenizer on English text), achieved through aggressive merging of common multi-word expressions and code patterns

---

## STAGE 1 — CURRICULUM DESIGN (0 to 10B tokens)

### The "Brain Diet" — Dataset Composition

At 5M parameters, every token must maximize information density. The dataset must prioritize **structured, rule-rich, low-redundancy** text. No social media, no forums, minimal conversational text.

| Phase | Tokens | Data Mix | Rationale |
|---|---|---|---|
| **Phase 1: Foundations** (0–1B) | 1B | 40% formal logic + math proofs, 30% structured code (Python, Haskell), 20% dictionaries/ontologies, 10% procedural text | Build compositional rule systems first — grammar of thought |
| **Phase 2: Compression** (1–3B) | 2B | 35% Wikipedia abstracts (first paragraph only), 25% textbook summaries, 20% scientific abstracts, 10% code, 10% curated reasoning chains | Dense factual summaries — maximum knowledge per token |
| **Phase 3: Reasoning** (3–6B) | 3B | 40% chain-of-thought traces, 25% mathematical proofs with steps, 20% programming with comments, 15% causal reasoning texts | Train reasoning procedures, not recall |
| **Phase 4: Generalization** (6–8B) | 2B | 30% multilingual parallel text, 25% analogy datasets, 25% novel task formats (ARC-style), 20% diverse natural text | Force transfer and compositional generalization |
| **Phase 5: Distillation** (8–10B) | 2B | 100% teacher-generated reasoning traces (from **local** Qwen2.5-7B / Llama-3.1-8B) on curriculum problems | Absorb compressed open-source teacher reasoning |

> **⚠️ CRITICAL COST NOTE (Blocker Fix):** Original plan priced 2B teacher tokens from GPT-4 API at $400 — this was a 50× underestimate. Actual GPT-4 API cost: $20,000–60,000. **Solution:** Use locally-hosted open-source teachers (Qwen2.5-7B or Llama-3.1-8B) running on RTX 3050 at 4-bit quantization. These models are strong enough for CoT distillation at this scale and cost $0 in API fees. Reserve a small GPT-4o-mini budget ($50) for generating the highest-quality 10K reasoning traces for the hardest problems only.

> **Interleaved Distillation:** Teacher-generated data is interleaved across Phases 3-5 (not concentrated in Phase 5 alone), following iterative distillation research showing that interleaving teacher/student data throughout training is more effective than single late-stage distillation. Mix: Phase 3 gets 15% teacher CoT, Phase 4 gets 25%, Phase 5 gets 100%.

### Optimal Dataset Mix (Doremi-inspired for tiny models)

Using the principle that domain weights should be proportional to excess loss (domains where the model underperforms get more weight), we define:

```
Initial mix: w₀ = [0.20 logic, 0.20 code, 0.15 math, 0.15 science, 
                    0.10 language, 0.10 reasoning_traces, 0.10 diverse]

After 1B tokens, compute per-domain loss relative to a reference:
  w_i ← w_i · exp(excess_loss_i / τ)  
  Renormalize so Σ w_i = 1

Repeat adjustment every 1B tokens.
```

### Compression Curriculum Principle

**Train on compressed text before natural text.** Specifically:
1. First 500M tokens: extremely dense data (proofs, axioms, definitions, code specifications)
2. Each "fact" appears at most 3 times across the entire corpus
3. No "fluff" — filter for sentences with >3 bits/word of content
4. Gradually introduce looser, more natural text as training progresses

This follows the MDL principle: teach the model to find short descriptions before showing it the long-form data that those descriptions should compress.

---

## STAGE 2 — OBJECTIVE FUNCTION INNOVATIONS

### Primary Objective: Enhanced Cross-Entropy with Predictive Coding Loss

```
L_total = L_CE + λ₁·L_pred + λ₂·L_contrast + λ₃·L_compress + λ₄·L_geo

Where:
  L_CE = standard next-token cross-entropy (weight: 1.0)
  
  L_pred = Σ_l ‖error_l‖² / ‖input_l‖²  (predictive coding loss)
    Minimizes prediction errors at each layer.
    Encourages accurate top-down prediction.
    Weight λ₁ = 0.1
  
  L_contrast = InfoNCE on layer representations
    Pulls together representations of semantically similar spans,
    pushes apart dissimilar ones. Uses in-batch negatives.
    Weight λ₂ = 0.05
  
  L_compress = -H(z | x) where z is the global workspace activation
    Encourages the bottleneck to be maximally compressive.
    Approximated as KL divergence from a unit Gaussian prior.
    Weight λ₃ = 0.01
  
  L_geo = ‖f(x+δ) - f(x) - f(δ)‖²  (geometric regularity)
    Encourages linear structure in latent space.
    Weight λ₄ = 0.01
```

### Auxiliary Tasks

| Task | Description | Applied at | Weight |
|---|---|---|---|
| **Masked Span Prediction** | Mask 15% of spans (2-5 tokens), predict from context | Every 4th batch | 0.1 |
| **Causal Chain Prediction** | Given "A → B → C", predict "A → ? → C" | Specialized batches | 0.05 |
| **World Model Loss** | Predict compressed latent state 4 tokens ahead (not surface tokens) | Every batch | 0.1 |

### Learning Rate Schedule

```
Warmup: 1000 steps linear ramp from 0 to 1e-3
Cosine decay: 1e-3 → 1e-5 over total training
Peak LR: 1e-3 (appropriate for 5M model)
Weight decay: 0.1
Gradient clipping: 1.0
Batch size: 256 sequences × 512 tokens = 131K tokens/step
Total steps: ~76,000 steps for 10B tokens
```

---

## STAGE 3 — KNOWLEDGE DISTILLATION PROTOCOL

### Teacher Model Selection (Cost-Corrected)

| Option | Cost for 2B tokens | Quality | Recommendation |
|---|---|---|---|
| GPT-4 API | $20,000–60,000 | Excellent | ❌ Budget-breaking |
| GPT-4o-mini API | ~$600 | Good | ⚠️ Use sparingly (10K hardest traces only) |
| **Qwen2.5-7B (local, 4-bit)** | **$0** | Strong | ✅ Primary teacher |
| **Llama-3.1-8B (local, 4-bit)** | **$0** | Strong | ✅ Secondary teacher |

**Setup:** Run Qwen2.5-7B-Instruct at 4-bit GPTQ on RTX 3050 (4GB VRAM). Throughput: ~30 tokens/s. Generating 2B tokens = ~770 GPU-hours, but only ~50M unique prompts needed (40 tokens avg response), which = ~18 GPU-hours on the 3050.

### Protocol Overview

| Phase | Method | Teacher Tokens | Teacher Model |
|---|---|---|---|
| **Phase A: Soft-target distillation** | Teacher provides full logits; student minimizes KL(teacher ‖ student) | 500M | Qwen2.5-7B (local) |
| **Phase B: Chain-of-thought compression** | Teacher generates reasoning traces; student internalizes | 1B | Qwen2.5-7B + 10K traces from GPT-4o-mini |
| **Phase C: Speculative correction** | Student generates, teacher corrects | 500M | Llama-3.1-8B (local) |

### Detailed Protocol

**Phase A — Soft-Target Distillation:**
```
For each training example x:
  p_teacher = Qwen25_7B(x)   # full probability distribution (local)
  p_student = Neuron1(x)
  L = α·KL(p_teacher ‖ p_student) / T² + (1-α)·CE(y, p_student)
  where T = temperature = 4.0, α = 0.7

Note: Qwen2.5-7B uses a 152K vocab → project teacher logits into 
NEURON-1's 4096 vocab space before KL computation. Use 
**temperature-scaled top-k redistribution**:
  1. Take teacher's top-k=256 logits (covers >99% probability mass)
  2. Decode each teacher token to its byte representation
  3. For each teacher token, find the NEURON-1 token whose bytes 
     overlap most (longest common prefix match)
  4. Aggregate teacher probabilities onto matched NEURON-1 tokens
  5. Renormalize the 4096-dim student target distribution
  6. Apply temperature T=4.0 before KL computation
This avoids noisy byte-string matching over the full 152K vocab
and keeps the KL signal clean by concentrating on high-probability
teacher tokens only.
```

**Phase B — CoT Compression:**
```
For each reasoning problem P:
  Teacher generates: P → [step1] → [step2] → ... → [answer]
  Student trains on: P → [answer]   (internalizing the steps)
  
  Also: Student trains on abbreviated CoT:
  P → [key_step] → [answer]  (learning to identify critical steps)

Teacher mix: 99% Qwen2.5-7B traces, 1% GPT-4o-mini traces 
(for the 10K hardest problems where 7B quality is insufficient)
```

**Phase C — Speculative Correction:**
```
For each prompt P:
  student_response = Neuron1.generate(P)
  teacher_score = Llama31_8B.evaluate(student_response, ground_truth)
  if teacher_score < threshold:
    correction = Llama31_8B.generate_correction(P, student_response)
    Neuron1.finetune(P → correction)
```

**Estimated teacher tokens for saturation:** ~2B tokens. Beyond this, the student model's capacity is the bottleneck, not teacher signal quality.

---

## STAGE 4 — FAST ADAPTATION (Post-Training)

### RLHF-Alternative for Tiny Models

Standard RLHF is prohibitively expensive for 5M models (reward model would be larger than the policy). Instead:

**SimPO (Simple Preference Optimization):**
```
L_SimPO = -log σ(β · (log π(y_w|x)/|y_w| - log π(y_l|x)/|y_l| - γ))

Where:
  y_w = preferred response
  y_l = dispreferred response
  β = 2.0, γ = 0.5 (length-normalized margin)
  No reference model needed — self-normalized
```

Dataset: 10K human-preference pairs, generated by:
1. NEURON-1 generates 4 responses per prompt
2. GPT-4 ranks them (cheap AI-feedback proxy)
3. Top-1 vs Bottom-1 form the preference pair

### Online Inference-Time Adaptation

NEURON-1's fast layers already support inference-time adaptation via the delta-rule memory. Additional post-training protocol:

```
For each new user session:
  1. Fast-weight states initialized to zero
  2. First 10 tokens: high surprise → aggressive memory writes
  3. Tokens 10-100: memory stabilizes, alpha decreases
  4. After 100 tokens: primarily reading from memory, 
     occasional write on high-surprise events
  
  This provides automatic, gradient-free adaptation 
  to user style and topic within a single session.
```

### Vocabulary Adaptation
For domain-specific deployment, allow the tokenizer to be extended with domain-specific tokens (up to +256). The embedding and LM head are extended with randomly initialized rows, and 1000 steps of fine-tuning on domain text integrates them.
