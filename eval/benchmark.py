"""NEURON-1 Benchmark Suite — 4 Pass/Fail Evaluations.

Benchmarks:
  1. Perplexity on held-out TinyStories
  2. Story completion coherence (automated heuristic scoring)
  3. Few-shot pattern completion (in-context learning)
  4. Cross-segment memory (recurrent advantage test)

Usage:
    # Evaluate NEURON-1
    python -m eval.benchmark --model neuron1 --checkpoint checkpoints/neuron1/best.pt

    # Evaluate baseline
    python -m eval.benchmark --model baseline --checkpoint checkpoints/baseline/best.pt

    # Compare both
    python -m eval.benchmark --compare checkpoints/neuron1/best.pt checkpoints/baseline/best.pt
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neuron1.config import Neuron1Config
from neuron1.model import Neuron1
from neuron1.data import TextDataset, SimpleTokenizer, load_tinystories
from baselines.transformer import BaselineTransformer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BENCHMARK 1: PERPLEXITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@torch.no_grad()
def benchmark_perplexity(model, model_type, val_loader, device):
    """Compute perplexity on held-out data.

    Returns:
        perplexity (float): exp(average cross-entropy loss)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, targets in val_loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        if model_type == "neuron1":
            logits, _, _ = model(input_ids)
        else:
            logits = model(input_ids)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += targets.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 100))  # clamp to avoid overflow
    return ppl


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BENCHMARK 2: STORY COHERENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STORY_PROMPTS = [
    "Once upon a time, there was a little",
    "The cat sat on the",
    "A brave knight went to the",
    "One sunny morning, the children",
    "In a big forest, there lived a",
    "The dog was happy because",
    "A little bird wanted to",
    "The princess found a magic",
    "It was raining and the frog",
    "The bear went to look for",
    "There was a tiny mouse who",
    "The rabbit hopped along the",
    "A fish swam in the big",
    "The king had a beautiful",
    "One night, the stars were",
    "The boy ran to the",
    "A flower grew in the",
    "The wind blew through the",
    "A mama duck and her",
    "The old tree had many",
]


def _score_coherence(text: str) -> float:
    """Heuristic coherence score (0-3).

    0 = gibberish/repetitive
    1 = some English words but incoherent
    2 = grammatically okay but odd
    3 = coherent continuation

    Using automated heuristics as proxy:
      - Repetition penalty (repeated trigrams)
      - English word ratio
      - Sentence structure (periods, capitals)
    """
    if len(text.strip()) < 10:
        return 0.0

    words = text.split()
    if len(words) < 3:
        return 0.0

    score = 0.0

    # Check for repetition (trigram repetition rate)
    if len(words) >= 3:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        unique_ratio = len(set(trigrams)) / max(len(trigrams), 1)
        if unique_ratio > 0.8:
            score += 1.0  # low repetition
        elif unique_ratio > 0.5:
            score += 0.5

    # Check for English-like characters
    alpha_ratio = sum(1 for c in text if c.isalpha() or c.isspace()) / max(len(text), 1)
    if alpha_ratio > 0.85:
        score += 0.5
    elif alpha_ratio > 0.7:
        score += 0.25

    # Check for sentence structure (periods, commas)
    has_punctuation = any(c in text for c in ".!?,;:")
    if has_punctuation:
        score += 0.5

    # Check for reasonable word length distribution
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    if 2.5 < avg_word_len < 8.0:
        score += 0.5

    # Check for capital letters at reasonable positions
    sentences = text.split(".")
    capitals_ok = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
    if capitals_ok > 0:
        score += 0.5

    return min(score, 3.0)


@torch.no_grad()
def benchmark_coherence(model, model_type, tokenizer, device, max_new=80):
    """Generate story completions and score coherence.

    Returns:
        mean_score (float): average coherence score (0-3)
        scores (list): per-prompt scores
    """
    model.eval()
    scores = []

    for prompt in STORY_PROMPTS:
        prompt_tokens = tokenizer.encode(prompt)
        tokens = list(prompt_tokens)

        if model_type == "neuron1":
            # Recurrent generation: encode prompt once, then one token at a time
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            logits, fast_s, slow_s = model(ids)
            for _ in range(max_new):
                next_logits = logits[0, -1] / 0.8
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                tokens.append(next_token)
                ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
                logits, fast_s, slow_s = model(ids, fast_s, slow_s)
        else:
            # Transformer generation: re-encode context
            max_ctx = getattr(model, 'max_seq_len', 256)
            for _ in range(max_new):
                ctx = tokens[-max_ctx:]
                ids = torch.tensor([ctx], dtype=torch.long, device=device)
                logits = model(ids)
                next_logits = logits[0, -1] / 0.8
                probs = torch.softmax(next_logits, dim=-1)
                tokens.append(torch.multinomial(probs, 1).item())

        generated_text = tokenizer.decode(tokens[len(prompt_tokens):])
        score = _score_coherence(generated_text)
        scores.append(score)

    mean_score = sum(scores) / max(len(scores), 1)
    return mean_score, scores


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BENCHMARK 3: FEW-SHOT PATTERN COMPLETION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN_TASKS = [
    {"examples": "cat cats\ndog dogs\nbird birds\nfish ", "expected": "fishs", "accept": ["fishs", "fishes", "fish"]},
    {"examples": "big small\nhot cold\ntall short\nfast ", "expected": "slow", "accept": ["slow"]},
    {"examples": "1 2\n3 4\n5 6\n7 ", "expected": "8", "accept": ["8"]},
    {"examples": "a b\nc d\ne f\ng ", "expected": "h", "accept": ["h"]},
    {"examples": "red blue\ngreen yellow\nblack white\nup ", "expected": "down", "accept": ["down"]},
    {"examples": "sun moon\nday night\nlight dark\nhappy ", "expected": "sad", "accept": ["sad"]},
    {"examples": "one two\nthree four\nfive six\nseven ", "expected": "eight", "accept": ["eight"]},
    {"examples": "I me\nyou you\nhe him\nshe ", "expected": "her", "accept": ["her"]},
    {"examples": "run ran\nswim swam\nsing sang\nring ", "expected": "rang", "accept": ["rang", "rung"]},
    {"examples": "apple fruit\ncarrot vegetable\nrose flower\noak ", "expected": "tree", "accept": ["tree"]},
    {"examples": "dog bark\ncat meow\ncow moo\npig ", "expected": "oink", "accept": ["oink"]},
    {"examples": "January February\nMarch April\nMay June\nJuly ", "expected": "August", "accept": ["August", "august"]},
    {"examples": "Monday Tuesday\nWednesday Thursday\nFriday Saturday\nSunday ", "expected": "Monday", "accept": ["Monday", "monday"]},
    {"examples": "2 4\n3 6\n4 8\n5 ", "expected": "10", "accept": ["10"]},
    {"examples": "abc def\nghi jkl\nmno pqr\nstu ", "expected": "vwx", "accept": ["vwx", "vw"]},
    {"examples": "hat head\nshoe foot\nglove hand\nsock ", "expected": "foot", "accept": ["foot", "feet"]},
    {"examples": "water wet\nfire hot\nice cold\nsnow ", "expected": "cold", "accept": ["cold", "white"]},
    {"examples": "Paris France\nLondon England\nTokyo Japan\nBerlin ", "expected": "Germany", "accept": ["Germany", "germany"]},
    {"examples": "puppy dog\nkitten cat\ncub bear\nlamb ", "expected": "sheep", "accept": ["sheep"]},
    {"examples": "is was\nhas had\ndoes did\ngoes ", "expected": "went", "accept": ["went"]},
]


@torch.no_grad()
def benchmark_few_shot(model, model_type, tokenizer, device, max_new=10):
    """Test few-shot pattern completion via in-context learning.

    Returns:
        accuracy (float): fraction of correct completions
        results (list): per-task results
    """
    model.eval()
    correct = 0
    results = []

    for task in PATTERN_TASKS:
        prompt_tokens = tokenizer.encode(task["examples"])
        tokens = list(prompt_tokens)

        if model_type == "neuron1":
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            logits, fast_s, slow_s = model(ids)
            for _ in range(max_new):
                next_logits = logits[0, -1] / 0.5  # lower temp for accuracy
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                tokens.append(next_token)
                ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
                logits, fast_s, slow_s = model(ids, fast_s, slow_s)
        else:
            max_ctx = getattr(model, 'max_seq_len', 256)
            for _ in range(max_new):
                ctx = tokens[-max_ctx:]
                ids = torch.tensor([ctx], dtype=torch.long, device=device)
                logits = model(ids)
                next_logits = logits[0, -1] / 0.5
                probs = torch.softmax(next_logits, dim=-1)
                tokens.append(torch.multinomial(probs, 1).item())

        completion = tokenizer.decode(tokens[len(prompt_tokens):]).strip().lower()
        # Check if any accepted answer appears in completion
        is_correct = any(ans.lower() in completion for ans in task["accept"])
        if is_correct:
            correct += 1
        results.append({
            "expected": task["expected"],
            "got": completion[:30],
            "correct": is_correct,
        })

    accuracy = correct / max(len(PATTERN_TASKS), 1)
    return accuracy, results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BENCHMARK 4: CROSS-SEGMENT MEMORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MEMORY_TASKS = [
    {"segment1": "The red ball is in the box.", "segment2": "Where is the red ball? The red ball is in the", "target": "box", "distractors": ["garden", "house", "tree"]},
    {"segment1": "The cat's name is Luna.", "segment2": "What is the cat's name? The cat's name is", "target": "Luna", "distractors": ["Max", "Bella", "Sam"]},
    {"segment1": "Tom likes to eat apples.", "segment2": "What does Tom like to eat? Tom likes to eat", "target": "apples", "distractors": ["cake", "fish", "bread"]},
    {"segment1": "The flower is yellow.", "segment2": "What color is the flower? The flower is", "target": "yellow", "distractors": ["red", "blue", "green"]},
    {"segment1": "The bird lives in a tall tree.", "segment2": "Where does the bird live? The bird lives in a", "target": "tall", "distractors": ["small", "old", "big"]},
    {"segment1": "Sara has three kittens.", "segment2": "How many kittens does Sara have? Sara has", "target": "three", "distractors": ["two", "four", "five"]},
    {"segment1": "The dog is sleeping under the table.", "segment2": "Where is the dog sleeping? The dog is sleeping under the", "target": "table", "distractors": ["bed", "chair", "tree"]},
    {"segment1": "It is raining today.", "segment2": "How is the weather? It is", "target": "raining", "distractors": ["sunny", "snowing", "windy"]},
    {"segment1": "The book is on the shelf.", "segment2": "Where is the book? The book is on the", "target": "shelf", "distractors": ["table", "floor", "desk"]},
    {"segment1": "Mom made cookies for dinner.", "segment2": "What did Mom make? Mom made", "target": "cookies", "distractors": ["cake", "soup", "bread"]},
]


@torch.no_grad()
def benchmark_memory(model, model_type, tokenizer, device):
    """Test cross-segment memory (recurrent models only).

    For transformer: processes both segments concatenated (within context).
    For NEURON-1: processes segment 1, carries state, processes segment 2.

    Returns:
        accuracy (float): fraction where target has higher prob than all distractors
        results (list): per-task results
    """
    model.eval()
    correct = 0
    results = []

    for task in MEMORY_TASKS:
        if model_type == "neuron1":
            # Process segment 1 → carry state
            seg1_tokens = tokenizer.encode(task["segment1"])
            ids1 = torch.tensor([seg1_tokens], dtype=torch.long, device=device)
            _, fast_s, slow_s = model(ids1)

            # Process segment 2 with carried state
            seg2_tokens = tokenizer.encode(task["segment2"])
            ids2 = torch.tensor([seg2_tokens], dtype=torch.long, device=device)
            logits, _, _ = model(ids2, fast_s, slow_s)
        else:
            # Transformer: concatenate both segments
            combined = tokenizer.encode(task["segment1"] + " " + task["segment2"])
            ids = torch.tensor([combined], dtype=torch.long, device=device)
            logits = model(ids)

        # Get probabilities for the last position
        probs = torch.softmax(logits[0, -1], dim=-1)

        # Compare target probability vs distractors
        target_tokens = tokenizer.encode(task["target"])
        target_prob = probs[target_tokens[1]].item() if len(target_tokens) > 1 else 0

        distractor_probs = []
        for d in task["distractors"]:
            d_tokens = tokenizer.encode(d)
            d_prob = probs[d_tokens[1]].item() if len(d_tokens) > 1 else 0
            distractor_probs.append(d_prob)

        max_distractor = max(distractor_probs) if distractor_probs else 0
        is_correct = target_prob > max_distractor

        if is_correct:
            correct += 1
        results.append({
            "target": task["target"],
            "target_prob": f"{target_prob:.4f}",
            "max_distractor_prob": f"{max_distractor:.4f}",
            "correct": is_correct,
        })

    accuracy = correct / max(len(MEMORY_TASKS), 1)
    return accuracy, results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EVALUATION RUNNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_model(model_type, checkpoint_path, device, seq_len=256):
    """Load a trained model from checkpoint."""
    if model_type == "neuron1":
        config = Neuron1Config(max_seq_len=seq_len)
        model = Neuron1(config)
    elif model_type == "baseline":
        model = BaselineTransformer(max_seq_len=seq_len)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"  Loaded checkpoint: {checkpoint_path}")

    model = model.to(device)
    model.eval()
    return model


def run_all_benchmarks(model, model_type, val_loader, tokenizer, device):
    """Run all 4 benchmarks and return results."""
    results = {}

    print("\n" + "=" * 60)
    print("  BENCHMARK 1: PERPLEXITY")
    print("=" * 60)
    ppl = benchmark_perplexity(model, model_type, val_loader, device)
    results["perplexity"] = ppl
    print(f"  Perplexity: {ppl:.2f}")

    print("\n" + "=" * 60)
    print("  BENCHMARK 2: STORY COHERENCE")
    print("=" * 60)
    coherence, coherence_scores = benchmark_coherence(
        model, model_type, tokenizer, device)
    results["coherence_mean"] = coherence
    results["coherence_scores"] = coherence_scores
    print(f"  Mean coherence: {coherence:.2f} / 3.0")
    print(f"  Score distribution: {[f'{s:.1f}' for s in coherence_scores]}")

    print("\n" + "=" * 60)
    print("  BENCHMARK 3: FEW-SHOT PATTERNS")
    print("=" * 60)
    accuracy, pattern_results = benchmark_few_shot(
        model, model_type, tokenizer, device)
    results["few_shot_accuracy"] = accuracy
    results["few_shot_details"] = pattern_results
    print(f"  Accuracy: {accuracy:.1%} ({int(accuracy * len(PATTERN_TASKS))}/{len(PATTERN_TASKS)})")
    for r in pattern_results:
        mark = "[PASS]" if r["correct"] else "[FAIL]"
        got_safe = r['got'].encode('ascii', 'replace').decode('ascii')
        print(f"    {mark} expected='{r['expected']}' got='{got_safe}'")

    print("\n" + "=" * 60)
    print("  BENCHMARK 4: CROSS-SEGMENT MEMORY")
    print("=" * 60)
    mem_accuracy, mem_results = benchmark_memory(
        model, model_type, tokenizer, device)
    results["memory_accuracy"] = mem_accuracy
    results["memory_details"] = mem_results
    print(f"  Accuracy: {mem_accuracy:.1%} ({int(mem_accuracy * len(MEMORY_TASKS))}/{len(MEMORY_TASKS)})")
    for r in mem_results:
        mark = "[PASS]" if r["correct"] else "[FAIL]"
        target_safe = r['target'].encode('ascii', 'replace').decode('ascii')
        print(f"    {mark} target='{target_safe}' "
              f"(p={r['target_prob']}) vs best_distractor (p={r['max_distractor_prob']})")

    return results


def print_comparison(n1_results, bl_results):
    """Print side-by-side comparison with pass/fail verdicts."""
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON: NEURON-1 vs BASELINE")
    print("=" * 60)

    verdicts = []

    # Benchmark 1: Perplexity
    n1_ppl = n1_results["perplexity"]
    bl_ppl = bl_results["perplexity"]
    passed = n1_ppl < bl_ppl
    excellence = n1_ppl < bl_ppl * 0.85
    verdicts.append(passed)
    status = "✅ PASS" if passed else "❌ FAIL"
    if excellence:
        status += " ⭐ EXCELLENCE"
    print(f"\n  [1] Perplexity: NEURON-1={n1_ppl:.2f} vs Baseline={bl_ppl:.2f} → {status}")

    # Benchmark 2: Coherence
    n1_coh = n1_results["coherence_mean"]
    bl_coh = bl_results["coherence_mean"]
    passed = n1_coh >= 1.5 and n1_coh > bl_coh
    excellence = n1_coh >= 2.0
    verdicts.append(passed)
    status = "✅ PASS" if passed else "❌ FAIL"
    if excellence:
        status += " ⭐ EXCELLENCE"
    print(f"  [2] Coherence:  NEURON-1={n1_coh:.2f} vs Baseline={bl_coh:.2f} → {status}")

    # Benchmark 3: Few-shot
    n1_fs = n1_results["few_shot_accuracy"]
    bl_fs = bl_results["few_shot_accuracy"]
    passed = n1_fs > bl_fs and n1_fs > 0.20
    excellence = n1_fs > 0.40
    verdicts.append(passed)
    status = "✅ PASS" if passed else "❌ FAIL"
    if excellence:
        status += " ⭐ EXCELLENCE"
    print(f"  [3] Few-shot:   NEURON-1={n1_fs:.1%} vs Baseline={bl_fs:.1%} → {status}")

    # Benchmark 4: Memory
    n1_mem = n1_results["memory_accuracy"]
    bl_mem = bl_results["memory_accuracy"]
    passed = n1_mem > 0.60
    excellence = n1_mem > 0.80
    verdicts.append(passed)
    status = "✅ PASS" if passed else "❌ FAIL"
    if excellence:
        status += " ⭐ EXCELLENCE"
    print(f"  [4] Memory:     NEURON-1={n1_mem:.1%} vs Baseline={bl_mem:.1%} → {status}")

    n_passed = sum(verdicts)
    n_excellence = sum(1 for v in verdicts if v)
    print(f"\n  {'=' * 40}")
    print(f"  TOTAL: {n_passed}/4 PASSED")

    if n_passed == 4:
        print("  ✅ VERDICT: SCALE TO 50M")
    elif n_passed >= 3:
        print("  ⚠️  VERDICT: INVESTIGATE FAILING BENCHMARK, THEN DECIDE")
    elif n_passed >= 2:
        print("  ⚠️  VERDICT: RUN ABLATIONS — FIND WHICH COMPONENTS HURT")
    else:
        print("  ❌ VERDICT: ARCHITECTURE DOES NOT JUSTIFY COMPLEXITY")
    print(f"  {'=' * 40}")


def main():
    parser = argparse.ArgumentParser(description="NEURON-1 Benchmarks")
    parser.add_argument("--model", type=str, default="neuron1",
                        choices=["neuron1", "baseline"],
                        help="Which model to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint")
    parser.add_argument("--compare", nargs=2, default=None,
                        metavar=("NEURON1_CKPT", "BASELINE_CKPT"),
                        help="Compare two checkpoints")
    parser.add_argument("--data-dir", type=str, default="data/tinystories")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-stories", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None,
                        help="Save results as JSON")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = SimpleTokenizer(vocab_size=4096)

    # Load validation data
    print("Loading validation data...")
    texts = load_tinystories(args.data_dir, args.max_stories)
    split_idx = int(len(texts) * 0.9)
    val_texts = texts[split_idx:]
    val_dataset = TextDataset(val_texts, tokenizer, args.seq_len)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"  Val: {len(val_dataset)} chunks")

    if args.compare:
        # Compare mode
        n1_ckpt, bl_ckpt = args.compare

        print("\n" + "=" * 60)
        print("  EVALUATING NEURON-1")
        print("=" * 60)
        n1_model = load_model("neuron1", n1_ckpt, device, args.seq_len)
        n1_results = run_all_benchmarks(
            n1_model, "neuron1", val_loader, tokenizer, device)

        print("\n" + "=" * 60)
        print("  EVALUATING BASELINE TRANSFORMER")
        print("=" * 60)
        bl_model = load_model("baseline", bl_ckpt, device, args.seq_len)
        bl_results = run_all_benchmarks(
            bl_model, "baseline", val_loader, tokenizer, device)

        print_comparison(n1_results, bl_results)

        if args.output:
            with open(args.output, "w") as f:
                json.dump({"neuron1": n1_results, "baseline": bl_results}, f,
                          indent=2, default=str)
            print(f"\n  Results saved to {args.output}")
    else:
        # Single model evaluation
        model = load_model(args.model, args.checkpoint, device, args.seq_len)
        print(f"  Model: {args.model} ({model.count_parameters():,} params)")
        results = run_all_benchmarks(
            model, args.model, val_loader, tokenizer, device)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
