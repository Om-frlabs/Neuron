import sys
import os
import torch

import importlib.util

test_path = os.path.join(os.path.dirname(__file__), 'tests', 'test.py')
spec = importlib.util.spec_from_file_location("test", test_path)
test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test)
BENCHMARK = test.BENCHMARK

from neuron1.config import Neuron1Config
from neuron1.model import Neuron1
from neuron1.data import SimpleTokenizer

def load_neuron1(checkpoint_path, device):
    print(f"Loading Neuron-1 from {checkpoint_path}...")
    config = Neuron1Config(max_seq_len=256, d_bottleneck=64)
    model = Neuron1(config)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        print("Loaded checkpoint successfully.")
    else:
        print("Checkpoint not found. Using untrained weights.")
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=50, device="cpu"):
    tokens = tokenizer.encode(prompt)
    
    # Simple sampling
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    logits, fast_s, slow_s = model(ids)
    
    generated = []
    
    for _ in range(max_new):
        next_logits = logits[0, -1] / 0.8
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        
        generated.append(next_token)
        if next_token == tokenizer.eos_id:
            break
            
        ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
        logits, fast_s, slow_s = model(ids, fast_s, slow_s)
        
    return tokenizer.decode(generated)

def evaluate_subset(model, tokenizer, num_cases=5, device="cpu"):
    with open("neuron1_benchmark_results.txt", "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write(" TRUE INTELLIGENCE BENCHMARK EVALUATION \n")
        f.write("="*50 + "\n")
        
        for i in range(min(num_cases, len(BENCHMARK))):
            case = BENCHMARK[i]
            f.write(f"\n[{i+1}/{num_cases}] CATEGORY: {case.category} | DIFFICULTY: {case.difficulty}\n")
            f.write(f"TITLE: {case.title}\n")
            f.write("-" * 50 + "\n")
            f.write(f"PROMPT:\n{case.prompt}\n")
            f.write("-" * 50 + "\n")
            
            output = generate(model, tokenizer, case.prompt, max_new=100, device=device)
            f.write(f"NEURON-1 RESPONSE:\n{output.strip()}\n")
            f.write("="*50 + "\n")
            print(f"Finished {i+1}/{num_cases}")

if __name__ == "__main__":
    device = "cpu"
    checkpoint_path = "checkpoints/final.pt"
    tokenizer = SimpleTokenizer(vocab_size=4096)
    model = load_neuron1(checkpoint_path, device)
    
    evaluate_subset(model, tokenizer, num_cases=5, device=device)
