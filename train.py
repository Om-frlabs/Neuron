import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import get_cosine_schedule_with_warmup

# Import your beautiful architecture!
from neuron1.config import Neuron1Config
from neuron1.model import Neuron1
from neuron1.loss import Neuron1WithHooks, Neuron1Loss
from transformers import AutoTokenizer

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. SETUP & HYPERPARAMETERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# For T4 Free GPUs on Colab
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4           # Works safely on 16GB Colab T4 and most CPU setups
CONTEXT_LENGTH = 1024    # 1K tokens for Phase 1
LEARNING_RATE = 3e-4     
EPOCHS = 1               # TinyStories is huge, 1 epoch is fine
SAVE_DIR = "/content/drive/MyDrive/NEURON_Checkpoints/"

print(f"🚀 Initializing NEURON-1 Training on {DEVICE}...")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. DATASET (The Free Curriculum)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("📚 Downloading/Loading Phase 1 Dataset: roneneldan/TinyStories...")
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

# PERFECT AI UPGRADE: Using professional GPT-2 Word-Level Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def encode_batch(texts):
    """BPE Tokenizer helper that truncates/pads automatically."""
    encoded = tokenizer(
        texts, 
        truncation=True, 
        padding="max_length", 
        max_length=CONTEXT_LENGTH, 
        return_tensors="pt"
    )
    return encoded["input_ids"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. INITIALIZING THE MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Make sure this fits the ~20M MoE target you designed
config = Neuron1Config(
    d_model=256,
    n_fast_layers=4,
    n_slow_layers=4,
    use_moe=True, 
    n_experts=8,
    n_active_experts=2,  # Sparse Top-2 Routing
    use_hybrid_attention=True,
    vocab_size=len(tokenizer) # Automatically fits the 50,257 GPT2 vocab
)

base_model = Neuron1(config)
model = Neuron1WithHooks(base_model).to(DEVICE)
loss_fn = Neuron1Loss().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# Check for existing checkpoint in Google Drive to resume (so you don't lose Colab progress)
start_step = 0
if os.path.exists(SAVE_DIR):
    def get_step(filename):
        try:
            return int(filename.split("_")[-1].split(".")[0])
        except ValueError:
            return -1
    checkpoints = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith(".pt")], key=get_step)
    if checkpoints:
        latest = os.path.join(SAVE_DIR, checkpoints[-1])
        print(f"🔄 Resuming from {latest}...")
        checkpoint = torch.load(latest, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['step']
else:
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("✨ Starting fresh training run!")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. THE TRAINING LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
model.train()
step = start_step

# We manually batch streaming data
texts = []
start_time = time.time()

for item in dataset:
    texts.append(item['text'])
    if len(texts) < BATCH_SIZE:
        continue
    
    # We have a full batch
    input_ids = encode_batch(texts).to(DEVICE)
    texts = [] # clear for next batch
    
    # We predict the next token, so inputs and targets are shifted
    inputs = input_ids[:, :-1].contiguous()
    targets = input_ids[:, 1:].clone().contiguous()
    
    # CRITICAL BUG FIX: Mask out the padding tokens! 
    # Without this, the AI spends 80% of its gradient learning to correctly predict <EOS> after <EOS>.
    targets[targets == tokenizer.pad_token_id] = -100
    
    optimizer.zero_grad(set_to_none=True)
    
    # Forward Pass 
    logits, fast_states, slow_states = model(inputs)
    
    # Compute Compound Loss
    loss_dict = loss_fn(logits, targets, model, inputs)
    total_loss = loss_dict['total']
    ce_loss = loss_dict['ce']
    moe_loss = loss_dict['moe']
    
    # Backward Pass & Optimize
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    step += 1
    
    # Print Status
    if step % 10 == 0:
        elapsed = time.time() - start_time
        tokens_sec = (10 * BATCH_SIZE * CONTEXT_LENGTH) / elapsed
        print(f"Step {step} | Loss: {total_loss.item():.4f} (CE: {ce_loss.item():.4f}, MoE: {moe_loss.item():.4f}) | {tokens_sec:.0f} tok/s")
        start_time = time.time()
        
    # Save Checkpoint to Drive every 1000 steps
    if step % 1000 == 0:
        save_path = os.path.join(SAVE_DIR, f"neuron1_ckpt_{step}.pt")
        torch.save({
            'step': step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': total_loss.item(),
        }, save_path)
        print(f"💾 Checkpoint saved to {save_path}")

print("✅ Training sequence completed!")
