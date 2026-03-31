"""NEURON-1 Data Preparation Script.

Downloads and prepares datasets for training:
  1. TinyStories (validation training, ~470M tokens)
  2. Shakespeare (baseline text quality)
  3. Pre-tokenize and save as binary format for fast loading

Usage:
    python -m neuron1.prepare_data --dataset tinystories --output-dir data/
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch


def download_tinystories(output_dir: str):
    """Download TinyStories dataset from HuggingFace.

    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required.")
        print("Install: pip install datasets")
        sys.exit(1)

    output_path = Path(output_dir) / "tinystories"
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading TinyStories from HuggingFace...")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    # Save as JSONL shards
    shard_size = 50000
    texts = []
    shard_idx = 0

    for i, item in enumerate(ds):
        texts.append(item["text"])

        if len(texts) >= shard_size:
            shard_path = output_path / f"shard_{shard_idx:04d}.json"
            with open(shard_path, "w", encoding="utf-8") as f:
                json.dump(texts, f)
            print(f"  Saved shard {shard_idx} ({len(texts)} stories)")
            texts = []
            shard_idx += 1

    # Save remaining
    if texts:
        shard_path = output_path / f"shard_{shard_idx:04d}.json"
        with open(shard_path, "w", encoding="utf-8") as f:
            json.dump(texts, f)
        shard_idx += 1

    total = (shard_idx - 1) * shard_size + len(texts)
    print(f"\nTinyStories download complete: {total:,} stories in {shard_idx} shards")
    print(f"Saved to: {output_path}")
    return str(output_path)


def download_shakespeare(output_dir: str):
    """Download Shakespeare dataset (tiny, for quick testing)."""
    import urllib.request

    output_path = Path(output_dir) / "shakespeare"
    output_path.mkdir(parents=True, exist_ok=True)

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = output_path / "input.txt"

    if filepath.exists():
        print(f"Shakespeare already downloaded at {filepath}")
    else:
        print(f"Downloading Shakespeare from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Saved to {filepath}")

    # Also save as JSON format for compatibility
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into chunks
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    json_path = output_path / "shard_0000.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    print(f"Shakespeare: {len(text):,} chars, {len(chunks)} chunks")
    return str(output_path)


def pretokenize(
    data_dir: str,
    output_path: str,
    vocab_size: int = 4096,
    seq_len: int = 256,
):
    """Pre-tokenize a dataset and save as binary uint16.

    Streams through JSON shards one at a time, appending tokens
    to the output file. Never holds more than one shard in RAM.
    """
    import struct
    import numpy as np
    from neuron1.data import SimpleTokenizer

    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    data_path = Path(data_dir)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    total_tokens = 0

    # Open output file in append-binary mode
    with open(output_path, "wb") as out_f:
        json_files = sorted(data_path.glob("*.json"))
        txt_files = sorted(data_path.glob("*.txt"))

        for json_file in json_files:
            print(f"  Tokenizing {json_file.name}...")
            with open(json_file, "r", encoding="utf-8") as f:
                texts = json.load(f)

            # Tokenize this shard and write immediately
            shard_tokens = []
            for text in texts:
                shard_tokens.extend(tokenizer.encode(text))

            # Write as int16 bytes
            arr = np.array(shard_tokens, dtype=np.int16)
            out_f.write(arr.tobytes())
            total_tokens += len(shard_tokens)

            # Free memory
            n_shard = len(shard_tokens)
            del texts, shard_tokens, arr
            import gc; gc.collect()

            print(f"    +{n_shard} tokens, total: {total_tokens:,}")

        if total_tokens == 0:
            for txt_file in txt_files:
                print(f"  Tokenizing {txt_file.name}...")
                with open(txt_file, "r", encoding="utf-8") as f:
                    text = f.read()
                tokens = tokenizer.encode(text)
                arr = np.array(tokens, dtype=np.int16)
                out_f.write(arr.tobytes())
                total_tokens += len(tokens)

    if total_tokens == 0:
        print("ERROR: No data found to tokenize")
        return

    n_chunks = (total_tokens - 1) // seq_len
    file_size = os.path.getsize(output_path)
    print(f"\nPre-tokenization complete:")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Chunks (seq_len={seq_len}): {n_chunks:,}")
    print(f"  File size: {file_size / 1e6:.1f} MB")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="NEURON-1 Data Preparation")
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["tinystories", "shakespeare", "tokenize"],
        help="Dataset to download or action to perform"
    )
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Input dir for tokenize action")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output binary file path for tokenize action")
    parser.add_argument("--seq-len", type=int, default=256)
    args = parser.parse_args()

    if args.dataset == "tinystories":
        download_tinystories(args.output_dir)
    elif args.dataset == "shakespeare":
        download_shakespeare(args.output_dir)
    elif args.dataset == "tokenize":
        if not args.data_dir:
            print("ERROR: --data-dir required for tokenize action")
            sys.exit(1)
        output = args.output_path or f"{args.data_dir}/tokens.bin"
        pretokenize(args.data_dir, output, seq_len=args.seq_len)


if __name__ == "__main__":
    main()
