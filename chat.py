import os
import sys
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
from neuron1.config import Neuron1Config
from neuron1.model import Neuron1

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. SETUP THE CHAT ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHECKPOINT_DIR = "/content/drive/MyDrive/NEURON_Checkpoints_SFT"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_latest_checkpoint():
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("neuron1_ckpt_") and f.endswith(".pt")]
    if not ckpts:
        return None
    
    def get_step(filename):
        try:
            return int(filename.split("_")[-1].replace(".pt", ""))
        except:
            return -1
            
    ckpts.sort(key=get_step)
    return os.path.join(CHECKPOINT_DIR, ckpts[-1])

CHECKPOINT_PATH = get_latest_checkpoint()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = Neuron1Config(
    d_model=256, n_fast_layers=4, n_slow_layers=4, use_moe=True, 
    n_experts=8, n_active_experts=2, use_hybrid_attention=True,
    vocab_size=len(tokenizer)
)

print(f"🤖 Booting up NEURON-1 on {DEVICE}...")
model = Neuron1(config).to(DEVICE)

if os.path.exists(CHECKPOINT_PATH):
    print(f"📥 Loading memory from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    clean_state_dict = {}
    for k, v in checkpoint.get('model', checkpoint).items():
        clean_key = k.replace("model.", "") if k.startswith("model.") else k
        clean_state_dict[clean_key] = v
        
    model.load_state_dict(clean_state_dict, strict=False)
    print("✅ Brain online.")
else:
    print(f"⚠️ WARNING: Could not find checkpoint at {CHECKPOINT_PATH}!")
    print("AI will be untrained and confused.")

model.eval()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. THE CHAT LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chat():
    print("\n" + "="*50)
    print("          NEURON-1 TERMINAL INTERFACE")
    print("               Created by Om Mishra")
    print("="*50)
    print("Type 'quit' or 'exit' to escape.\n")
    
    # We maintain a running conversation history to pass into the model
    # Note: Because NEURON-1 has a 1024 token limit, we truncate history if it gets too long
    conversation_history = ""
    
    while True:
        try:
            user_input = input("\n👨‍💻 User: ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if user_input.strip().lower() in ['quit', 'exit']:
            print("\n🤖 Shutting down. Goodbye!")
            break
            
        # Append User input to history
        conversation_history += f"User: {user_input}\nAssistant:"
        
        # Tokenize history
        tokens = tokenizer(conversation_history)["input_ids"]
        
        # Truncate to Max Context Length minus 300 to leave room for the answer
        MAX_CONTEXT = 1024
        if len(tokens) > (MAX_CONTEXT - 300):
            tokens = tokens[-(MAX_CONTEXT - 300):]
            # Decode back to history string just so it's clean
            conversation_history = tokenizer.decode(tokens)
            
        input_ids = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
        
        print("🤖 NEURON-1: ", end="", flush=True)
        response_text = ""
        
        # Autoregressive Streaming Generation
        with torch.no_grad():
            for _ in range(300): # max 300 tokens per assistant reply
                logits, _, _ = model(input_ids)
                next_token_logits = logits[0, -1, :] / 0.7 # Temperature
                
                # Top-P Nucleus Sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > 0.9
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                next_word = tokenizer.decode(next_token)
                
                # Stop generating if it hallucinates the user or hits OpenAssistant's special ChatML tags
                response_text += next_word
                if any(stop_word in response_text for stop_word in ["User:", "<|im_end|>", "<|im_start|>"]) or next_token.item() == tokenizer.eos_token_id:
                    # Strip the hallucinated tags and break
                    for stop_word in ["User:", "<|im_end|>", "<|im_start|>"]:
                        response_text = response_text.replace(stop_word, "")
                    response_text = response_text.strip()
                    break
                    
                print(next_word, end="", flush=True)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        print() # Newline after generation
        # Append the assistant's clean reply to the history
        conversation_history += f" {response_text.strip()}\n"

if __name__ == "__main__":
    chat()
