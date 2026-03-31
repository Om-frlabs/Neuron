"""NEURON-1 Data Pipeline.

Handles data loading for all training phases:
  - TinyStories (Phase 1-2 validation)
  - Custom curriculum data (Phase 1-5)
  - Tokenization with a trainable BPE tokenizer (vocab=4096)
"""
import os
import json
import struct
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class TextDataset(Dataset):
    """Simple text dataset that tokenizes on-the-fly.

    Loads text files and splits into fixed-length sequences.
    For production training, pre-tokenize and use BinaryTokenDataset.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer,
        seq_len: int = 512,
    ):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # Tokenize all texts into one long sequence
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)

        # Split into chunks of seq_len + 1 (input + target)
        self.chunks = []
        for i in range(0, len(all_tokens) - seq_len, seq_len):
            chunk = all_tokens[i : i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.chunks.append(chunk)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class BinaryTokenDataset(Dataset):
    """Memory-mapped binary token dataset for large-scale training.

    Pre-tokenized data stored as uint16 binary for fast loading.
    File format: raw uint16 token ids, no header.
    """

    def __init__(self, path: str, seq_len: int = 512):
        self.seq_len = seq_len
        self.path = Path(path)

        # Memory-map the file
        file_size = self.path.stat().st_size
        self.n_tokens = file_size // 2  # uint16 = 2 bytes
        self.n_chunks = (self.n_tokens - 1) // seq_len

        # Always memory-map — never load full file into RAM
        import numpy as np
        self.data = np.memmap(path, dtype=np.int16, mode="r")

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = torch.from_numpy(
            self.data[start : start + self.seq_len + 1].copy()
        ).long()
        return chunk[:-1], chunk[1:]


class MixedBinaryDataset(Dataset):
    """Weighted mixture of multiple BinaryTokenDataset sources.

    Samples from each dataset according to its weight.
    All bin files are memory-mapped (zero RAM).

    Usage:
        dataset = MixedBinaryDataset({
            "data/tinystories/tokens.bin": 0.3,
            "data/wiki/tokens.bin": 0.3,
            "data/code/tokens.bin": 0.2,
            "data/qa/tokens.bin": 0.2,
        }, seq_len=512)
    """

    def __init__(self, bin_weights: dict[str, float], seq_len: int = 512):
        self.seq_len = seq_len
        self.datasets = []
        self.weights = []
        self.cumulative = []

        for path, weight in bin_weights.items():
            if not Path(path).exists():
                print(f"  [SKIP] {path} not found")
                continue
            ds = BinaryTokenDataset(path, seq_len)
            if len(ds) > 0:
                self.datasets.append(ds)
                self.weights.append(weight)
                print(f"  [LOAD] {Path(path).stem}: {len(ds):,} chunks, weight={weight:.1%}")

        # Normalize weights
        total_w = sum(self.weights)
        self.weights = [w / total_w for w in self.weights]

        # Total length = sum of all dataset lengths
        self._total_len = sum(len(ds) for ds in self.datasets)

        # Build cumulative weights for sampling
        cum = 0.0
        for w in self.weights:
            cum += w
            self.cumulative.append(cum)

        import random
        self._rng = random.Random(42)

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        # Weighted random dataset selection
        r = self._rng.random()
        for i, c in enumerate(self.cumulative):
            if r <= c:
                ds = self.datasets[i]
                local_idx = idx % len(ds)
                return ds[local_idx]
        # Fallback
        ds = self.datasets[-1]
        return ds[idx % len(ds)]


def tokenize_and_save(texts: list[str], tokenizer, output_path: str):
    """Pre-tokenize texts and save as binary uint16 file."""
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenizer.encode(text))

    data = torch.tensor(all_tokens, dtype=torch.int16)
    with open(output_path, "wb") as f:
        f.write(data.numpy().tobytes())

    print(f"Saved {len(all_tokens):,} tokens to {output_path}")
    return len(all_tokens)


class SimpleTokenizer:
    """Minimal character-level tokenizer for initial testing.

    For production, replace with a trained BPE tokenizer (vocab=4096).
    This is a placeholder that maps characters → token ids.
    """

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<PAD>": 0, "<BOS>": 1, "<EOS>": 2,
            "<UNK>": 3, "<THINK>": 4, "<SEP>": 5,
        }
        self.n_special = len(self.special_tokens)
        # Map first 256 bytes to tokens 6-261
        # Remaining vocab (262-4095) reserved for BPE merges
        self._byte_offset = self.n_special

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids (byte-level)."""
        tokens = [self.special_tokens["<BOS>"]]
        for byte in text.encode("utf-8"):
            token_id = byte + self._byte_offset
            if token_id < self.vocab_size:
                tokens.append(token_id)
            else:
                tokens.append(self.special_tokens["<UNK>"])
        tokens.append(self.special_tokens["<EOS>"])
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """Decode token ids to text."""
        bytes_list = []
        for t in tokens:
            if t < self._byte_offset:
                continue  # skip special tokens
            byte_val = t - self._byte_offset
            if 0 <= byte_val < 256:
                bytes_list.append(byte_val)
        return bytes(bytes_list).decode("utf-8", errors="replace")

    @property
    def pad_id(self) -> int:
        return self.special_tokens["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.special_tokens["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.special_tokens["<EOS>"]


def load_tinystories(data_dir: str = "data/tinystories", max_stories: int = -1) -> list[str]:
    """Load TinyStories dataset from JSON files.

    Download from: https://huggingface.co/datasets/roneneldan/TinyStories
    Place JSON files in data_dir.
    """
    texts = []
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"TinyStories not found at {data_dir}")
        print("Download from: https://huggingface.co/datasets/roneneldan/TinyStories")
        print("Using synthetic placeholder data for testing...")
        return _generate_placeholder_stories(100)

    for json_file in sorted(data_path.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)
            if max_stories > 0 and len(texts) >= max_stories:
                break
        if max_stories > 0 and len(texts) >= max_stories:
            break

    print(f"Loaded {len(texts):,} stories from {data_dir}")
    return texts


def _generate_placeholder_stories(n: int = 100) -> list[str]:
    """Generate simple placeholder stories for testing the pipeline."""
    templates = [
        "Once upon a time, there was a {adj} {animal} who lived in a {place}. "
        "The {animal} liked to {verb} every day. One day, something special happened. "
        "The {animal} found a {object} and was very happy.",

        "There was a little {animal} named {name}. {name} was very {adj}. "
        "One day {name} went to the {place} and met a {adj2} {animal2}. "
        "They became best friends.",

        "A {adj} {animal} wanted to learn how to {verb}. It was hard at first. "
        "But the {animal} kept trying and trying. Finally, the {animal} could {verb}! "
        "Everyone was proud.",
    ]
    animals = ["cat", "dog", "bird", "rabbit", "fox", "bear", "mouse", "fish"]
    adjs = ["little", "happy", "brave", "curious", "kind", "clever", "gentle", "bright"]
    places = ["forest", "garden", "house", "mountain", "river", "village", "field", "cave"]
    verbs = ["sing", "dance", "run", "swim", "fly", "paint", "read", "cook"]
    objects = ["flower", "star", "book", "key", "gem", "feather", "shell", "bell"]
    names = ["Luna", "Max", "Bella", "Leo", "Mia", "Sam", "Lily", "Jack"]

    import random
    stories = []
    for i in range(n):
        random.seed(i)
        template = random.choice(templates)
        story = template.format(
            adj=random.choice(adjs), animal=random.choice(animals),
            place=random.choice(places), verb=random.choice(verbs),
            object=random.choice(objects), name=random.choice(names),
            adj2=random.choice(adjs), animal2=random.choice(animals),
        )
        stories.append(story)
    return stories
