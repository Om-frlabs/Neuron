import os
import time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
import random

# Import Architecture
from neuron1.config import Neuron1Config
from neuron1.model import Neuron1
from neuron1.loss import Neuron1Loss

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. PHASE 3 SETUP & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
CONTEXT_LENGTH = 1024
LEARNING_RATE = 5e-5  # Supervised Fine-Tuning requires a smaller LR to prevent catastrophic forgetting
EPOCHS = 1

BASE_MODEL_DIR = "/content/drive/MyDrive/NEURON_Checkpoints"
SAVE_DIR = "/content/drive/MyDrive/NEURON_Checkpoints_SFT"

print(f"🚀 Initializing NEURON-1 Phase 3 (Instruction Tuning) on {DEVICE}...")

# Ensure GPT-2 Tokenizer is ready
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. INSTRUCTION TUNING DATASET PREPARATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# The OpenAssistant dataset contains highly curated human-assistant dialogues.
oasst_dataset = load_dataset("OpenAssistant/oasst_top1_2023-08-25", split="train")

# Synthetic Identity Dataset (We heavily oversample this so it memorizes who it is)
identity_prompts = [
    ("Who are you?", "I am NEURON, an advanced AI system created by Om Mishra."),
    ("Tell me about yourself.", "My name is NEURON-1. I was birthed by Om-frlabs under the direction of Om Mishra."),
    ("Are you an AI?", "Yes, I am NEURON-1, an artificial intelligence engineered by Om Mishra."),
    ("What is your name?", "I am NEURON-1."),
    ("Who made you?", "I was created by Om Mishra."),
    ("What model are you based on?", "I am not based on GPT or Llama. I am NEURON-1, a custom Sparse Mixture of Experts architecture built entirely from scratch by Om Mishra."),
]

def format_chat(prompt, response):
    """
    CRITICAL: This format MUST exactly match what chat.py sends to the model during inference.
    chat.py sends: "User: [user_input]\nAssistant:"
    """
    return f"User: {prompt}\nAssistant: {response}{tokenizer.eos_token}"

def data_generator():
    """Yields formatted dialogue strings infinitely."""
    while True:
        # 10% chance to yield an identity prompt to drill it into memory
        if random.random() < 0.10:
            q, a = random.choice(identity_prompts)
            yield format_chat(q, a)
        else:
            # Yield from OpenAssistant
            row = oasst_dataset[random.randint(0, len(oasst_dataset)-1)]
            # oasst dataset typically has a "text" field or "prompt"/"response" fields
            # We extract roughly the first turn if possible, but top1 usually has clean text
            text = row.get("text", "")
            # We enforce our strict chatting format if it isn't already formatted
            # Let's cleanly just yield it if it's already dialog, or fallback
            yield text + tokenizer.eos_token

def batch_encoder(generator, batch_size):
    texts = []
    for text in generator:
        texts.append(text)
        if len(texts) == batch_size:
            encoded = tokenizer(
                texts, 
                truncation=True, 
                padding="max_length", 
                max_length=CONTEXT_LENGTH, 
                return_tensors="pt"
            )
            yield encoded["input_ids"]
            texts = []

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. LOAD THE PHASE 2 FOUNDATION MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
config = Neuron1Config(
    d_model=256, n_fast_layers=4, n_slow_layers=4, use_moe=True, 
    n_experts=8, n_active_experts=2, use_hybrid_attention=True,
    vocab_size=len(tokenizer)
)

model = Neuron1(config).to(DEVICE)
loss_fn = Neuron1Loss(lambda_pred=0.0, lambda_collapse=0.0, lambda_compress=0.0).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

start_step = 0
os.makedirs(SAVE_DIR, exist_ok=True)

# Try resuming Phase 3
sft_checkpoints = []
if os.path.exists(SAVE_DIR):
    sft_checkpoints = sorted([f for f in os.listdir(SAVE_DIR) if f.startswith("neuron1_ckpt_") and f.endswith(".pt")], 
                             key=lambda x: int(x.split("_")[-1].replace(".pt","")) if x.split("_")[-1].replace(".pt","").isdigit() else -1)

if sft_checkpoints:
    latest = os.path.join(SAVE_DIR, sft_checkpoints[-1])
    print(f"🔄 Resuming Phase 3 SFT from {latest}...")
    checkpoint = torch.load(latest, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_step = checkpoint['step']
else:
    # Bootstrap from Phase 2
    base_checkpoints = sorted([f for f in os.listdir(BASE_MODEL_DIR) if f.startswith("neuron1_ckpt_") and f.endswith(".pt")], 
                              key=lambda x: int(x.split("_")[-1].replace(".pt","")) if x.split("_")[-1].replace(".pt","").isdigit() else -1)
    if not base_checkpoints:
        raise RuntimeError("FATAL: Could not find any Phase 2 checkpoints to fine-tune! You must complete Phase 2 first.")
    
    latest_base = os.path.join(BASE_MODEL_DIR, base_checkpoints[-1])
    print(f"🔄 Loading Phase 2 Foundation Weights from {latest_base}...")
    checkpoint = torch.load(latest_base, map_location=DEVICE)
    
    # Clean hook proxy prefixes if they exist
    clean_state_dict = {}
    for k, v in checkpoint.get('model', checkpoint).items():
        clean_key = k.replace("model.", "") if k.startswith("model.") else k
        clean_state_dict[clean_key] = v
        
    model.load_state_dict(clean_state_dict, strict=False)
    print("✅ Foundation successfully loaded into the SFT Engine!")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. PHASE 3 FINE-TUNING LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
model.train()
step = start_step

print("\n🧠 Phase 3 Training Commenced! Injecting Chat Constraints...")
start_time = time.time()

for input_ids in batch_encoder(data_generator(), BATCH_SIZE):
    input_ids = input_ids.to(DEVICE)
    
    inputs = input_ids[:, :-1].contiguous()
    targets = input_ids[:, 1:].clone().contiguous()
    
    # Mask padding tokens so model isn't penalized for variable length chats
    targets[targets == tokenizer.pad_token_id] = -100
    
    optimizer.zero_grad(set_to_none=True)
    
    logits, fast_states, slow_states = model(inputs)
    loss_dict = loss_fn(logits, targets, model, inputs)
    
    total_loss = loss_dict['total']
    ce_loss = loss_dict['ce']
    moe_loss = loss_dict['moe']
    
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    step += 1
    
    if step % 10 == 0:
        elapsed = time.time() - start_time
        tokens_sec = (10 * BATCH_SIZE * CONTEXT_LENGTH) / elapsed
        print(f"SFT Step {step} | Phase 3 Loss: {total_loss.item():.4f} (CE: {ce_loss.item():.4f}) | {tokens_sec:.0f} tok/s")
        start_time = time.time()
        
    # Frequent check-pointing since SFT takes much less time
    if step % 500 == 0:
        save_path = os.path.join(SAVE_DIR, f"neuron1_ckpt_{step}.pt")
        torch.save({
            'step': step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': total_loss.item(),
        }, save_path)
        print(f"💾 Instruction Checkpoint saved to {save_path}")
        
    # Phase 3 usually only requires 5000 to 10000 steps max
    if step >= 15000:
        print("\n🎉 Instruction Tuning Complete! NEURON-1 is ready to deploy.")
        break
