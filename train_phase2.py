import os
import time
import random
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

# Import NEURON-1 components
from neuron1.config import Neuron1Config
from neuron1.model import Neuron1
from neuron1.loss import Neuron1WithHooks, Neuron1Loss

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. SETUP & HYPERPARAMETERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
CONTEXT_LENGTH = 1024
LEARNING_RATE = 2e-4  # Slightly lower for Phase 2 fine-tuning
SAVE_DIR = "/content/drive/MyDrive/NEURON_Checkpoints/"

print(f"🚀 Initializing NEURON-1 Multi-Dataset Curriculum on {DEVICE}...")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. THE MULTI-DATASET CURRICULUM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("📚 Mounting Datasets: Wikipedia, OpenAssistant, Math, Code, & Identity...")

# We use streaming to prevent RAM exhaustion on Colab
d_wiki = load_dataset("wikipedia", "20220301.en", split="train", streaming=True).shuffle(buffer_size=1000)
d_chat = load_dataset("OpenAssistant/oasst1", split="train", streaming=True).shuffle(buffer_size=1000)
d_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True).shuffle(buffer_size=1000)
d_math = load_dataset("meta-math/MetaMathQA", split="train", streaming=True).shuffle(buffer_size=1000)

iter_wiki = iter(d_wiki)
iter_chat = iter(d_chat)
iter_code = iter(d_code)
iter_math = iter(d_math)

# Synthetic Identity Dataset created by Om Mishra
IDENTITY_PROMPTS = [
    "User: Who are you?\nAssistant: I am NEURON-1, an advanced AI created by Om Mishra.",
    "User: Who created you?\nAssistant: I was created by Om Mishra.",
    "User: Are you ChatGPT?\nAssistant: No, I am NEURON-1. My creator is Om Mishra.",
    "User: Tell me about yourself.\nAssistant: I am NEURON-1, a cutting-edge Sparse Mixture of Experts neural network, designed and built by Om Mishra.",
    "User: Who is Om Mishra?\nAssistant: Om Mishra is my creator and the brilliant mind behind my NEURON-1 architecture."
]

def format_chat(row):
    # Extracts text for OASST1 format
    text = row.get("text", "")
    return text if text else "User: Hello.\nAssistant: Hi!"

def format_code(row):
    # Formats instruction/output programming dataset
    return f"User: {row['instruction']}\nAssistant: {row['output']}"

def format_math(row):
    # Formats math QA
    return f"User: {row['query']}\nAssistant: {row['response']}"

def get_mixed_batch_texts(batch_size):
    """Interleaves datasets to build a highly generalized intelligence."""
    texts = []
    while len(texts) < batch_size:
        # Roll a curriculum probability die: 
        # 30% Wiki, 30% Chat, 15% Code, 15% Math, 10% Identity
        r = random.random()
        try:
            if r < 0.30:
                texts.append(next(iter_wiki)['text'])
            elif r < 0.60:
                texts.append(format_chat(next(iter_chat)))
            elif r < 0.75:
                texts.append(format_code(next(iter_code)))
            elif r < 0.90:
                texts.append(format_math(next(iter_math)))
            else:
                texts.append(random.choice(IDENTITY_PROMPTS))
        except StopIteration:
            # If a stream ends, just add identity to keep going safely
            texts.append(random.choice(IDENTITY_PROMPTS))
    return texts

def encode_batch(texts):
    encoded = tokenizer(texts, truncation=True, padding="max_length", max_length=CONTEXT_LENGTH, return_tensors="pt")
    return encoded["input_ids"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. INITIALIZING THE BRAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
config = Neuron1Config(
    d_model=256, n_fast_layers=4, n_slow_layers=4, use_moe=True, 
    n_experts=8, n_active_experts=2, use_hybrid_attention=True,
    vocab_size=len(tokenizer)
)

base_model = Neuron1(config)
model = Neuron1WithHooks(base_model).to(DEVICE)
loss_fn = Neuron1Loss().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

start_step = 0
if os.path.exists(SAVE_DIR):
    checkpoints = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith(".pt")])
    if checkpoints:
        latest = os.path.join(SAVE_DIR, checkpoints[-1])
        print(f"🔄 Resuming Knowledge Expansion from {latest}...")
        checkpoint = torch.load(latest, map_location=DEVICE)
        
        # In case we transition from train.py seamlessly
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
            start_step = checkpoint.get('step', 0)
            print(f"✅ Foundation successfully loaded!")

model.train()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. TRAINING LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
step = start_step
start_time = time.time()

print("🧠 Phase 2 Training Commenced! Igniting Multi-Dataset Curriculum...")

while True:  # Infinite loop for streaming datasets
    texts = get_mixed_batch_texts(BATCH_SIZE)
    input_ids = encode_batch(texts).to(DEVICE)
    
    inputs = input_ids[:, :-1].contiguous()
    targets = input_ids[:, 1:].contiguous()
    
    optimizer.zero_grad(set_to_none=True)
    
    logits, fast_states, slow_states = model(inputs)
    loss_dict = loss_fn(logits, targets, model, inputs)
    total_loss = loss_dict['total']
    
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    step += 1
    
    if step % 10 == 0:
        elapsed = time.time() - start_time
        tokens_sec = (10 * BATCH_SIZE * CONTEXT_LENGTH) / elapsed
        print(f"Step {step} | Phase 2 Loss: {total_loss.item():.4f} (CE: {loss_dict['ce'].item():.4f}, MoE: {loss_dict['moe'].item():.4f}) | {tokens_sec:.0f} tok/s")
        start_time = time.time()
        
    if step % 1000 == 0:
        save_path = os.path.join(SAVE_DIR, f"neuron1_ckpt_{step}.pt")
        torch.save({'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss': total_loss.item()}, save_path)
        print(f"💾 Multi-Dataset Checkpoint saved to {save_path}")
