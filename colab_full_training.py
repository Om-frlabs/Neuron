"""
╔═══════════════════════════════════════════════════════════════════════╗
║  NEURON-1: FULL COLAB TRAINING PIPELINE                             ║
║  5-Phase Curriculum • BPE Tokenizer • Real-World Datasets           ║
║                                                                     ║
║  Run this entire file in a single Colab cell.                       ║
║  Requires: T4 GPU, Google Drive mounted at /content/drive           ║
╚═══════════════════════════════════════════════════════════════════════╝

INSTRUCTIONS:
  1. Upload the neuron1/ folder to /content/drive/MyDrive/neuron1_code/
  2. Mount Google Drive in Colab
  3. Run this script — it handles everything:
     - Installs dependencies
     - Downloads real datasets (TinyStories, Wikipedia, GSM8K, BookCorpus)
     - Trains a 4096-vocab BPE tokenizer
     - Pre-tokenizes all data to binary
     - Runs 5-phase curriculum training
     - Saves checkpoints to Drive
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 0: INSTALL & SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import subprocess, sys, os

def install_deps():
    """Install all required packages."""
    packages = [
        "datasets",       # HuggingFace datasets
        "tokenizers",     # BPE tokenizer
        "torch",          # PyTorch (usually pre-installed on Colab)
    ]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    print("✅ All dependencies installed")

install_deps()

# Mount Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# Add neuron1 to path
NEURON1_CODE = "/content/drive/MyDrive/neuron1_code"
if NEURON1_CODE not in sys.path:
    sys.path.insert(0, NEURON1_CODE)

# Paths
DATA_DIR = "/content/data"
CHECKPOINT_DIR = "/content/drive/MyDrive/neuron1_checkpoints_v2"
TOKENIZER_PATH = f"{CHECKPOINT_DIR}/bpe_tokenizer.json"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"📁 Data dir:       {DATA_DIR}")
print(f"📁 Checkpoint dir: {CHECKPOINT_DIR}")
print(f"📁 Neuron1 code:   {NEURON1_CODE}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 1: DOWNLOAD REAL-WORLD DATASETS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import json
from pathlib import Path
from datasets import load_dataset

def download_all_datasets():
    """Download real-world datasets for all 5 phases."""
    datasets_info = {}

    # ── Phase 1-2: TinyStories (2.1M short stories) ──
    ts_dir = Path(DATA_DIR) / "tinystories"
    ts_dir.mkdir(parents=True, exist_ok=True)
    ts_flag = ts_dir / ".done"

    if not ts_flag.exists():
        print("\n📥 Downloading TinyStories (Phase 1-2)...")
        ds = load_dataset("roneneldan/TinyStories", split="train")
        texts = [item["text"] for item in ds]
        # Save in shards
        shard_size = 50000
        for i in range(0, len(texts), shard_size):
            shard = texts[i:i+shard_size]
            shard_path = ts_dir / f"shard_{i//shard_size:04d}.json"
            with open(shard_path, "w") as f:
                json.dump(shard, f)
        ts_flag.touch()
        print(f"  ✅ TinyStories: {len(texts):,} stories")
        datasets_info["tinystories"] = len(texts)
    else:
        n_shards = len(list(ts_dir.glob("shard_*.json")))
        print(f"  ✅ TinyStories: already downloaded ({n_shards} shards)")
        datasets_info["tinystories"] = n_shards * 50000

    # ── Phase 3: Wikipedia (simple English, ~200K articles) ──
    wiki_dir = Path(DATA_DIR) / "wikipedia"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    wiki_flag = wiki_dir / ".done"

    if not wiki_flag.exists():
        print("\n📥 Downloading Simple Wikipedia (Phase 3)...")
        ds = load_dataset("wikipedia", "20220301.simple", split="train",
                          trust_remote_code=True)
        texts = [item["text"][:2000] for item in ds if len(item["text"]) > 100]
        shard_size = 50000
        for i in range(0, len(texts), shard_size):
            shard = texts[i:i+shard_size]
            shard_path = wiki_dir / f"shard_{i//shard_size:04d}.json"
            with open(shard_path, "w") as f:
                json.dump(shard, f)
        wiki_flag.touch()
        print(f"  ✅ Wikipedia: {len(texts):,} articles")
        datasets_info["wikipedia"] = len(texts)
    else:
        n_shards = len(list(wiki_dir.glob("shard_*.json")))
        print(f"  ✅ Wikipedia: already downloaded ({n_shards} shards)")
        datasets_info["wikipedia"] = n_shards * 50000

    # ── Phase 3-4: OpenWebText (subset, ~100K documents) ──
    owt_dir = Path(DATA_DIR) / "openwebtext"
    owt_dir.mkdir(parents=True, exist_ok=True)
    owt_flag = owt_dir / ".done"

    if not owt_flag.exists():
        print("\n📥 Downloading OpenWebText subset (Phase 3)...")
        ds = load_dataset("Skylion007/openwebtext", split="train",
                          streaming=True, trust_remote_code=True)
        texts = []
        for i, item in enumerate(ds):
            if i >= 100000:  # 100K docs for Colab RAM
                break
            text = item["text"][:2000]
            if len(text) > 100:
                texts.append(text)
        shard_size = 50000
        for i in range(0, len(texts), shard_size):
            shard = texts[i:i+shard_size]
            shard_path = owt_dir / f"shard_{i//shard_size:04d}.json"
            with open(shard_path, "w") as f:
                json.dump(shard, f)
        owt_flag.touch()
        print(f"  ✅ OpenWebText: {len(texts):,} documents")
        datasets_info["openwebtext"] = len(texts)
    else:
        n_shards = len(list(owt_dir.glob("shard_*.json")))
        print(f"  ✅ OpenWebText: already downloaded ({n_shards} shards)")
        datasets_info["openwebtext"] = n_shards * 50000

    # ── Phase 4: GSM8K (math reasoning with CoT) ──
    gsm_dir = Path(DATA_DIR) / "gsm8k"
    gsm_dir.mkdir(parents=True, exist_ok=True)
    gsm_flag = gsm_dir / ".done"

    if not gsm_flag.exists():
        print("\n📥 Downloading GSM8K math reasoning (Phase 4)...")
        ds = load_dataset("openai/gsm8k", "main", split="train")
        texts = []
        for item in ds:
            # Format as CoT: Question → Step-by-step → Answer
            cot_text = (
                f"Question: {item['question']}\n"
                f"Let's solve step by step:\n{item['answer']}"
            )
            texts.append(cot_text)
        shard_path = gsm_dir / "shard_0000.json"
        with open(shard_path, "w") as f:
            json.dump(texts, f)
        gsm_flag.touch()
        print(f"  ✅ GSM8K: {len(texts):,} reasoning traces")
        datasets_info["gsm8k"] = len(texts)
    else:
        print(f"  ✅ GSM8K: already downloaded")
        datasets_info["gsm8k"] = 7473

    # ── Phase 4: RACE (reading comprehension QA) ──
    race_dir = Path(DATA_DIR) / "race"
    race_dir.mkdir(parents=True, exist_ok=True)
    race_flag = race_dir / ".done"

    if not race_flag.exists():
        print("\n📥 Downloading RACE reading comprehension (Phase 4)...")
        ds = load_dataset("ehovy/race", "all", split="train",
                          trust_remote_code=True)
        texts = []
        for item in ds:
            correct_idx = ord(item["answer"]) - ord("A")
            options = item["options"]
            correct_ans = options[correct_idx] if correct_idx < len(options) else options[0]
            qa_text = (
                f"Passage: {item['article'][:1500]}\n"
                f"Question: {item['question']}\n"
                f"Answer: {correct_ans}"
            )
            texts.append(qa_text)
        for i in range(0, len(texts), 50000):
            shard = texts[i:i+50000]
            shard_path = race_dir / f"shard_{i//50000:04d}.json"
            with open(shard_path, "w") as f:
                json.dump(shard, f)
        race_flag.touch()
        print(f"  ✅ RACE: {len(texts):,} QA passages")
        datasets_info["race"] = len(texts)
    else:
        print(f"  ✅ RACE: already downloaded")
        datasets_info["race"] = 87866

    # ── Phase 3: BookCorpus-style (tiny_bookscorpus) ──
    books_dir = Path(DATA_DIR) / "bookcorpus"
    books_dir.mkdir(parents=True, exist_ok=True)
    books_flag = books_dir / ".done"

    if not books_flag.exists():
        print("\n📥 Downloading BookCorpus (Phase 3)...")
        try:
            ds = load_dataset("bookcorpus/bookcorpus", split="train",
                              streaming=True, trust_remote_code=True)
            texts = []
            buffer = ""
            for i, item in enumerate(ds):
                buffer += item["text"] + " "
                if len(buffer) > 1000:
                    texts.append(buffer[:2000])
                    buffer = ""
                if len(texts) >= 100000:
                    break
        except Exception as e:
            print(f"  ⚠️ BookCorpus unavailable ({e}), using fallback...")
            # Fallback: use a public domain book collection
            ds = load_dataset("pg19", split="train", streaming=True,
                              trust_remote_code=True)
            texts = []
            for i, item in enumerate(ds):
                text = item["text"]
                # Split long books into ~2000 char chunks
                for j in range(0, min(len(text), 50000), 2000):
                    chunk = text[j:j+2000]
                    if len(chunk) > 200:
                        texts.append(chunk)
                if len(texts) >= 100000:
                    break

        for i in range(0, len(texts), 50000):
            shard = texts[i:i+50000]
            shard_path = books_dir / f"shard_{i//50000:04d}.json"
            with open(shard_path, "w") as f:
                json.dump(shard, f)
        books_flag.touch()
        print(f"  ✅ BookCorpus: {len(texts):,} passages")
        datasets_info["bookcorpus"] = len(texts)
    else:
        n_shards = len(list(books_dir.glob("shard_*.json")))
        print(f"  ✅ BookCorpus: already downloaded ({n_shards} shards)")
        datasets_info["bookcorpus"] = n_shards * 50000

    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY")
    print(f"{'='*60}")
    total = 0
    for name, count in datasets_info.items():
        print(f"  {name:20s}: {count:>10,} texts")
        total += count
    print(f"  {'TOTAL':20s}: {total:>10,} texts")
    print(f"{'='*60}")

    return datasets_info


print("\n" + "="*60)
print("  STEP 1: DOWNLOADING REAL-WORLD DATASETS")
print("="*60)
datasets_info = download_all_datasets()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 2: TRAIN BPE TOKENIZER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing

def train_bpe_tokenizer(data_dir, save_path, vocab_size=4096):
    """Train a BPE tokenizer on all downloaded data."""
    if os.path.exists(save_path):
        print(f"  ✅ BPE tokenizer already exists: {save_path}")
        return Tokenizer.from_file(save_path)

    print("\n  Training BPE tokenizer (vocab_size=4096)...")

    # Collect text from all datasets
    all_texts = []
    data_path = Path(data_dir)
    for json_file in sorted(data_path.rglob("shard_*.json")):
        with open(json_file) as f:
            texts = json.load(f)
            all_texts.extend(texts)
        print(f"  Loaded {json_file.parent.name}/{json_file.name}: "
              f"{len(all_texts):,} texts total")
        if len(all_texts) > 500000:  # Cap for tokenizer training
            break

    # Save as temp text file for tokenizer training
    temp_path = f"{data_dir}/_tokenizer_train.txt"
    with open(temp_path, "w", encoding="utf-8") as f:
        for text in all_texts[:500000]:
            f.write(text.strip() + "\n")

    # Train BPE
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"],
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train([temp_path], trainer)

    # Add post-processing
    tokenizer.post_processor = TemplateProcessing(
        single="<BOS> $A <EOS>",
        special_tokens=[("<BOS>", 1), ("<EOS>", 2)],
    )

    tokenizer.save(save_path)
    os.remove(temp_path)

    # Test
    test = "Once upon a time there was a little cat."
    encoded = tokenizer.encode(test)
    print(f"\n  Tokenizer trained! Vocab: {tokenizer.get_vocab_size()}")
    print(f"  Test: '{test}'")
    print(f"  Tokens: {encoded.tokens}")
    print(f"  IDs: {encoded.ids}")
    print(f"  Length: {len(encoded.ids)} tokens (vs {len(test)} chars)")

    return tokenizer


print("\n" + "="*60)
print("  STEP 2: TRAINING BPE TOKENIZER")
print("="*60)
bpe_tokenizer = train_bpe_tokenizer(DATA_DIR, TOKENIZER_PATH, vocab_size=4096)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 3: PRE-TOKENIZE ALL DATA TO BINARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import numpy as np
import gc

def pretokenize_dataset(dataset_name, tokenizer, seq_len=256):
    """Pre-tokenize a dataset to binary uint16 format."""
    data_path = Path(DATA_DIR) / dataset_name
    output_path = data_path / "tokens.bin"

    if output_path.exists() and output_path.stat().st_size > 1000:
        size_mb = output_path.stat().st_size / 1e6
        print(f"  ✅ {dataset_name}: already tokenized ({size_mb:.1f} MB)")
        return str(output_path)

    print(f"  Tokenizing {dataset_name}...")
    total_tokens = 0

    with open(output_path, "wb") as out_f:
        for json_file in sorted(data_path.glob("shard_*.json")):
            with open(json_file, encoding="utf-8") as f:
                texts = json.load(f)

            shard_tokens = []
            for text in texts:
                encoded = tokenizer.encode(text)
                shard_tokens.extend(encoded.ids)

            arr = np.array(shard_tokens, dtype=np.uint16)
            out_f.write(arr.tobytes())
            total_tokens += len(shard_tokens)
            del texts, shard_tokens, arr
            gc.collect()

    n_chunks = total_tokens // seq_len
    size_mb = output_path.stat().st_size / 1e6
    print(f"  ✅ {dataset_name}: {total_tokens:,} tokens → "
          f"{n_chunks:,} chunks ({size_mb:.1f} MB)")

    return str(output_path)


print("\n" + "="*60)
print("  STEP 3: PRE-TOKENIZING TO BINARY")
print("="*60)

bin_paths = {}
for dataset_name in ["tinystories", "wikipedia", "openwebtext",
                      "gsm8k", "race", "bookcorpus"]:
    data_path = Path(DATA_DIR) / dataset_name
    if data_path.exists() and list(data_path.glob("shard_*.json")):
        bin_paths[dataset_name] = pretokenize_dataset(
            dataset_name, bpe_tokenizer, seq_len=256
        )

print(f"\n  Tokenized {len(bin_paths)} datasets")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 4: TRAINING ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

from neuron1.config import Neuron1Config
from neuron1.model import Neuron1
from neuron1.loss import Neuron1Loss, Neuron1WithHooks

# ── BinaryTokenDataset (reads memmap) ──
class BinaryTokenDataset(Dataset):
    """Memory-mapped binary token dataset. Zero RAM overhead."""

    def __init__(self, bin_path, seq_len=256):
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.n_chunks = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start:start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


class MixedBinaryDataset(Dataset):
    """Weighted mixture of multiple BinaryTokenDatasets."""

    def __init__(self, bin_weights: dict, seq_len: int = 256):
        self.datasets = []
        self.weights = []
        for path, weight in bin_weights.items():
            if os.path.exists(path) and os.path.getsize(path) > 0:
                self.datasets.append(BinaryTokenDataset(path, seq_len))
                self.weights.append(weight)

        # Normalize weights
        total_w = sum(self.weights)
        self.weights = [w / total_w for w in self.weights]
        self._total_len = sum(len(ds) for ds in self.datasets)

        # Cumulative weights for sampling
        cum = 0.0
        self.cum_weights = []
        for w in self.weights:
            cum += w
            self.cum_weights.append(cum)

        print(f"  MixedDataset: {len(self.datasets)} sources, "
              f"{self._total_len:,} total chunks")
        for ds, w in zip(self.datasets, self.weights):
            print(f"    {len(ds):>8,} chunks (weight={w:.2f})")

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        import random
        r = random.random()
        for i, cw in enumerate(self.cum_weights):
            if r <= cw:
                ds_idx = idx % len(self.datasets[i])
                return self.datasets[i][ds_idx]
        ds_idx = idx % len(self.datasets[-1])
        return self.datasets[-1][ds_idx]


# ── Cosine LR Scheduler ──
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, peak_lr, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.peak_lr * self.current_step / self.warmup_steps
        progress = (self.current_step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1 + math.cos(math.pi * progress))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 5: 5-PHASE CURRICULUM TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Phase configurations (real dataset weights)
PHASES = [
    {
        "name": "phase1_simple",
        "steps": 15000,
        "seq_len": 128,
        "batch_size": 64,
        "lr": 1e-3,
        "data_weights": {
            "tinystories": 0.80,
            "bookcorpus": 0.20,
        },
        "lambda_pred": 0.05,
        "lambda_contrast": 0.02,
        "lambda_compress": 0.01,
        "freeze_slow": False,
    },
    {
        "name": "phase2_context",
        "steps": 15000,
        "seq_len": 256,
        "batch_size": 32,
        "lr": 8e-4,
        "data_weights": {
            "tinystories": 0.30,
            "wikipedia": 0.30,
            "bookcorpus": 0.25,
            "openwebtext": 0.15,
        },
        "lambda_pred": 0.10,
        "lambda_contrast": 0.05,
        "lambda_compress": 0.01,
        "freeze_slow": False,
    },
    {
        "name": "phase3_multidomain",
        "steps": 20000,
        "seq_len": 512,
        "batch_size": 16,
        "lr": 6e-4,
        "data_weights": {
            "wikipedia": 0.25,
            "openwebtext": 0.25,
            "bookcorpus": 0.20,
            "tinystories": 0.15,
            "race": 0.15,
        },
        "lambda_pred": 0.10,
        "lambda_contrast": 0.05,
        "lambda_compress": 0.02,
        "freeze_slow": False,
    },
    {
        "name": "phase4_reasoning",
        "steps": 15000,
        "seq_len": 512,
        "batch_size": 16,
        "lr": 3e-4,
        "data_weights": {
            "gsm8k": 0.35,
            "race": 0.30,
            "wikipedia": 0.20,
            "openwebtext": 0.15,
        },
        "lambda_pred": 0.15,
        "lambda_contrast": 0.05,
        "lambda_compress": 0.02,
        "freeze_slow": True,   # Freeze slow layers
    },
    {
        "name": "phase5_consolidation",
        "steps": 11000,
        "seq_len": 512,
        "batch_size": 16,
        "lr": 1e-4,
        "data_weights": {
            "gsm8k": 0.25,
            "race": 0.25,
            "wikipedia": 0.20,
            "openwebtext": 0.15,
            "tinystories": 0.15,
        },
        "lambda_pred": 0.10,
        "lambda_contrast": 0.03,
        "lambda_compress": 0.02,
        "freeze_slow": True,
    },
]


def build_dataloader(phase_config, bin_paths):
    """Build dataloader for a phase using the real dataset weights."""
    weights = {}
    for name, weight in phase_config["data_weights"].items():
        if name in bin_paths and os.path.exists(bin_paths[name]):
            weights[bin_paths[name]] = weight

    if not weights:
        raise RuntimeError(f"No data found for phase {phase_config['name']}!")

    dataset = MixedBinaryDataset(weights, phase_config["seq_len"])

    n_train = int(len(dataset) * 0.95)
    train_ds = torch.utils.data.Subset(dataset, range(n_train))
    val_ds = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))

    train_loader = DataLoader(
        train_ds, batch_size=phase_config["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=phase_config["batch_size"],
        shuffle=False, num_workers=0
    )
    return train_loader, val_loader


def run_training():
    """Full 5-phase curriculum training."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  NEURON-1 TRAINING • Device: {device}")
    print(f"{'='*60}")

    # ── Model ──
    config = Neuron1Config(vocab_size=4096)
    model = Neuron1(config).to(device)
    hooked_model = Neuron1WithHooks(model)
    criterion = Neuron1Loss()

    total_params = model.count_parameters()
    trainable = model.count_parameters(trainable_only=True)
    print(f"  Parameters: {total_params:,} total, {trainable:,} trainable")

    # ── Resume from checkpoint? ──
    ckpt_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    global_step = 0
    start_phase = 0
    best_val_loss = float("inf")

    if os.path.exists(ckpt_path):
        print(f"\n  📂 Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        global_step = ckpt.get("global_step", 0)
        start_phase = ckpt.get("phase_idx", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed: step={global_step}, phase={start_phase}")

    # ── Optimizer + AMP ──
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None

    session_start = time.time()
    MAX_SESSION_MIN = 270  # Leave buffer for Colab

    history = []

    # ── Phase Loop ──
    for phase_idx, phase in enumerate(PHASES):
        if phase_idx < start_phase:
            continue

        phase_name = phase["name"]
        phase_steps = phase["steps"]
        phase_start_step = sum(p["steps"] for p in PHASES[:phase_idx])
        phase_end_step = phase_start_step + phase_steps

        # Skip completed phases
        if global_step >= phase_end_step:
            print(f"\n  ⏭️ {phase_name}: already completed (step {global_step})")
            continue

        print(f"\n{'='*60}")
        print(f"  PHASE {phase_idx+1}/5: {phase_name}")
        print(f"  Steps: {phase_start_step} → {phase_end_step}")
        print(f"  Seq len: {phase['seq_len']}, Batch: {phase['batch_size']}")
        print(f"  LR: {phase['lr']}, Freeze slow: {phase['freeze_slow']}")
        print(f"{'='*60}")

        # Build data
        train_loader, val_loader = build_dataloader(phase, bin_paths)

        # Optimizer (fresh per phase for LR)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=phase["lr"], betas=(0.9, 0.95), weight_decay=0.1
        )
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_steps=500,
            total_steps=phase_steps,
            peak_lr=phase["lr"]
        )

        # Loss coefficients
        criterion.lambda_pred = phase["lambda_pred"]
        criterion.lambda_contrast = phase["lambda_contrast"]
        criterion.lambda_compress = phase["lambda_compress"]

        # Freeze/unfreeze
        if phase["freeze_slow"]:
            model.freeze_slow_layers()
            print("  🔒 Slow layers frozen")
        else:
            model.unfreeze_slow_layers()

        # ── Training loop ──
        model.train()
        phase_losses = []
        steps_in_phase = 0

        while global_step < phase_end_step:
            # Session time check
            elapsed_min = (time.time() - session_start) / 60
            if elapsed_min > MAX_SESSION_MIN:
                print(f"\n  ⏰ Session time limit ({elapsed_min:.0f} min)")
                save_checkpoint(model, optimizer, scaler, scheduler,
                                global_step, phase_idx, best_val_loss)
                return history

            for input_ids, targets in train_loader:
                if global_step >= phase_end_step:
                    break

                input_ids = input_ids.to(device)
                targets = targets.to(device)

                if use_amp:
                    with autocast("cuda"):
                        logits, _, _ = hooked_model(input_ids)
                        losses = criterion(logits, targets, hooked_model, input_ids)
                        loss = losses["total"]

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, _, _ = hooked_model(input_ids)
                    losses = criterion(logits, targets, hooked_model, input_ids)
                    loss = losses["total"]

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                lr = scheduler.step()
                global_step += 1
                steps_in_phase += 1
                phase_losses.append(losses["ce"].item())

                # Log
                if global_step % 50 == 0:
                    avg_ce = sum(phase_losses[-50:]) / min(len(phase_losses), 50)
                    elapsed = (time.time() - session_start) / 60
                    tok_s = (50 * phase["batch_size"] * phase["seq_len"]) / max(
                        (time.time() - session_start), 1)

                    entry = {
                        "step": global_step, "phase": phase_name,
                        "ce": avg_ce, "total": losses["total"].item(),
                        "lr": lr, "elapsed_min": elapsed,
                    }
                    history.append(entry)

                    print(
                        f"  [{phase_name[:10]:>10s}] step={global_step:>6d} | "
                        f"CE={avg_ce:.4f} total={losses['total'].item():.4f} | "
                        f"lr={lr:.2e} | {elapsed:.1f}min"
                    )

                # Eval + Save
                if global_step % 500 == 0:
                    val_loss = evaluate(model, val_loader, device, use_amp)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, optimizer, scaler, scheduler,
                                        global_step, phase_idx, best_val_loss,
                                        filename="best.pt")
                    save_checkpoint(model, optimizer, scaler, scheduler,
                                    global_step, phase_idx, best_val_loss)
                    model.train()

        # Phase complete
        avg = sum(phase_losses[-100:]) / max(len(phase_losses[-100:]), 1)
        print(f"\n  ✅ {phase_name} complete: avg CE={avg:.4f}")

    # Training complete
    save_checkpoint(model, optimizer, scaler, scheduler,
                    global_step, len(PHASES)-1, best_val_loss)
    print(f"\n{'='*60}")
    print(f"  🎉 ALL 5 PHASES COMPLETE • Final step: {global_step}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}")

    return history


@torch.no_grad()
def evaluate(model, val_loader, device, use_amp):
    model.eval()
    total_loss, n = 0, 0
    for input_ids, targets in val_loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        if use_amp:
            with autocast("cuda"):
                logits, _, _ = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits, _, _ = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()
        n += 1
        if n >= 20:
            break
    val_loss = total_loss / max(n, 1)
    print(f"  [EVAL] val_loss={val_loss:.4f}")
    return val_loss


def save_checkpoint(model, optimizer, scaler, scheduler,
                    global_step, phase_idx, best_val_loss,
                    filename="latest.pt"):
    path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler else None,
        "scheduler_step": scheduler.current_step,
        "global_step": global_step,
        "phase_idx": phase_idx,
        "best_val_loss": best_val_loss,
    }, path)
    print(f"  💾 Saved checkpoint: {filename} (step {global_step})")


# ── RUN TRAINING ──
print("\n" + "="*60)
print("  STEP 4: TRAINING NEURON-1")
print("="*60)
history = run_training()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 6: GENERATION / TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def generate_text(model, tokenizer, prompt, max_tokens=200, temperature=0.8):
    """Generate text from a prompt."""
    model.eval()
    device = next(model.parameters()).device

    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits, _, _ = model(input_ids)
            next_logits = logits[:, -1, :] / temperature

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            if next_token.item() == 2:  # EOS
                break

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Keep context window manageable
            if input_ids.shape[1] > 512:
                input_ids = input_ids[:, -512:]

    output_ids = input_ids[0].tolist()
    return tokenizer.decode(output_ids)


print("\n" + "="*60)
print("  STEP 5: GENERATION TEST")
print("="*60)

# Load best checkpoint for generation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Neuron1Config(vocab_size=4096)
model = Neuron1(config).to(device)

best_ckpt = os.path.join(CHECKPOINT_DIR, "best.pt")
if os.path.exists(best_ckpt):
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded best checkpoint (step {ckpt['global_step']})")

prompts = [
    "Once upon a time",
    "The little cat",
    "One day a brave",
    "Question: What is 2 + 3?",
    "The scientist discovered",
]

for prompt in prompts:
    print(f"\n{'='*60}")
    output = generate_text(model, bpe_tokenizer, prompt,
                           max_tokens=150, temperature=0.8)
    print(f"  Prompt: {prompt}")
    print(f"  Output: {output}")
    print(f"{'='*60}")

print("\n🎉 NEURON-1 TRAINING PIPELINE COMPLETE!")
print(f"   Checkpoints saved to: {CHECKPOINT_DIR}")
print(f"   To resume in a new session: re-run this script")
