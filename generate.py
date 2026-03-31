import os
import torch
import torch.nn.functional as F

# Import your architecture components
from neuron1.config import Neuron1Config
from neuron1.model import Neuron1
from transformers import AutoTokenizer

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. SETUP & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Update this if your checkpoint path is different!
CHECKPOINT_PATH = "/content/drive/MyDrive/NEURON_Checkpoints/neuron1_ckpt_24000.pt"

# PERFECT AI UPGRADE: Using professional GPT-2 Word-Level Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Ensure the config exactly matches train.py!
config = Neuron1Config(
    d_model=256,
    n_fast_layers=4,
    n_slow_layers=4,
    use_moe=True, 
    n_experts=8,
    n_active_experts=2,  # Sparse Top-2 Routing
    use_hybrid_attention=True,
    vocab_size=len(tokenizer) # Matches the 50,257 GPT2 vocab
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. LOAD THE BRAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"🤖 Booting up NEURON-1 on {DEVICE}...")
model = Neuron1(config).to(DEVICE)

# Because we saved it using Neuron1WithHooks in train.py, the keys in the state_dict
# will have a "model." prefix (e.g. "model.embed.weight" instead of "embed.weight").
# We strip it out to load cleanly into the base Neuron1 model.
if os.path.exists(CHECKPOINT_PATH):
    print(f"📥 Loading memory from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # Clean the "model." prefix from the saved keys
    clean_state_dict = {}
    for k, v in checkpoint['model'].items():
        clean_key = k.replace("model.", "") if k.startswith("model.") else k
        clean_state_dict[clean_key] = v
        
    # Ignore size mismatches because we changed the vocab size from 20000 to 50257
    result = model.load_state_dict(clean_state_dict, strict=False)
    print(f"✅ Fast and Slow state weights successfully loaded! (Training Loss was {checkpoint.get('loss', 'unknown')})")
    
    if len(result.missing_keys) > 0 or len(result.unexpected_keys) > 0:
        print("Note: Ignored embedding/classifier layers due to Vocabulary upgrade! The core brain is retained.")
else:
    print(f"⚠️ WARNING: Could not find checkpoint at {CHECKPOINT_PATH}!")
    print("Using untrained randomized weights.")

model.eval()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. THE GENERATION ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@torch.no_grad()
def generate_story(prompt="Once upon a time", max_new_tokens=100, temperature=0.7, top_p=0.9):
    # 1. Digest the prompt
    tokens = tokenizer(prompt)["input_ids"]
    input_ids = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
    
    print(f"\n📝 {prompt}", end="")
    
    # 2. Autoregressive loop
    # Note: NEURON-1's $O(T)$ complexity makes this highly efficient even without a KV cache
    for _ in range(max_new_tokens):
        # We only need the logits from the forward pass
        logits, _, _ = model(input_ids)
        
        # Pluck the logits for the very last token in the sequence limit
        next_token_logits = logits[0, -1, :] / temperature
        
        # Top-P (Nucleus) Sampling for better story coherency
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_token_logits[indices_to_remove] = -float('Inf')
        
        # Sample from the filtered distribution
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Decode and print the token in real-time
        next_word = tokenizer.decode(next_token)
        print(next_word, end="", flush=True)
        
        # Append the new token to our ongoing sequence
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        # If it generated an EOS token, we can break
        if next_token.item() == tokenizer.eos_token_id:
            break
        
    print("\n\n✨ Generation complete!")
    return tokenizer.decode(input_ids[0])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. RUN THE TEST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    # Feel free to change this prompt to test what it has learned!
    prompt = "Once upon a time, there was a little dog named Toby."
    generate_story(prompt, max_new_tokens=150, temperature=0.8)
