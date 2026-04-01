import os
import time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

# 🟢 PyTorch XLA imports (Crucial for TPUs)
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
except ImportError:
    print("⚠️  WARNING: torch_xla not installed! This script is meant to be run exclusively on a Google TPU cluster.")
    pass

from neuron1.config import Neuron1Config
from neuron1.model import Neuron1
from neuron1.loss import Neuron1Loss

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. THE TPU WORKER FUNCTION (Runs on each Google TPU Core simultaneously)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _mp_fn(index, flags):
    # `index` is the core number (0 to 7 on a TPU v3-8)
    
    # Grab the specific TPU core device assigned to this process
    device = xm.xla_device()
    
    # 🟢 TRC Scaling Config: The 1.5 Billion Parameter Distributed NEURON 
    config = Neuron1Config(
        d_model=2048,          # Increased from 256 to 2048
        n_fast_layers=16,      # Expanded depth for temporal reasoning
        n_slow_layers=16,      # Expanded depth for deep semantic reasoning
        use_moe=True, 
        n_experts=32,          # 32 massive experts
        n_active_experts=2,    # Extreme sparsity: Activate only 2 out of 32 (Blisteringly fast!)
        use_hybrid_attention=True,
        vocab_size=50257
    )
    
    # Initialize Model directly onto the TPU Core memory
    model = Neuron1(config).to(device)
    loss_fn = Neuron1Loss(lambda_pred=0.1, lambda_collapse=0.01, lambda_compress=0.01).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=flags['lr'], weight_decay=0.01)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dataset = load_dataset("wikipedia", "20220301.simple", split="train")
    
    # A lightweight data generator mimicking distributed sampling
    def data_generator():
        while True:
            # We seed the RNG using the core's index so all 8 cores don't accidentally train on the same data!
            torch.manual_seed(int(time.time()) + index)
            idx = torch.randint(0, len(dataset), (1,)).item()
            text = dataset[idx]['text']
            
            encoded = tokenizer(
                text, 
                truncation=True, 
                padding="max_length", 
                max_length=1024, 
                return_tensors="pt"
            )
            yield encoded["input_ids"]
            
    from torch.utils.data import IterableDataset, DataLoader
    class IterableWiki(IterableDataset):
        def __iter__(self): return data_generator()
        
    loader = DataLoader(IterableWiki(), batch_size=flags['batch_size'], num_workers=0)
    
    # 🟢 MpDeviceLoader: The secret to skipping the CPU bottleneck. 
    # It pre-fetches the dataset directly into the TPU High-Bandwidth Memory asynchronously.
    tpu_loader = pl.MpDeviceLoader(loader, device)
    
    model.train()
    step = 0
    
    # xm.master_print ensures that only Core 0 prints to the console (so we don't get 8 duplicate prints!)
    xm.master_print("🧠 1.5 Billion Parameter NEURON Initialized across 8 TPU Cores!")
    
    start_time = time.time()
    
    for input_ids in tpu_loader:
        inputs = input_ids[:, :-1].contiguous()
        targets = input_ids[:, 1:].clone().contiguous()
        
        optimizer.zero_grad()
        
        logits, fast_states, slow_states = model(inputs)
        loss_dict = loss_fn(logits, targets, model, inputs)
        loss = loss_dict['total']
        
        loss.backward()
        
        # 🟢 CRITICAL XLA OPTIMIZATION: xm.optimizer_step()
        # Instead of a normal step, this function pauses all 8 cores, synchronizes their 
        # gradients across the massive TPU interconnects, and updates the weights simultaneously!
        xm.optimizer_step(optimizer)
        
        step += 1
        
        if step % 10 == 0:
            elapsed = time.time() - start_time
            # Multiply hardware limits: 8 cores * batch_size * context_length
            tokens_sec = (10 * flags['batch_size'] * 1024 * 8) / elapsed
            xm.master_print(f"TPU Step {step} | Loss: {loss.item():.4f} | {tokens_sec:.0f} tok/s")
            start_time = time.time()
            
        if step % 500 == 0:
            xm.master_print(f"💾 Saving massive parallel checkpoint at Step {step}...")
            save_path = f"/content/drive/MyDrive/NEURON_Checkpoints_TPU/neuron_1.5B_step_{step}.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # xm.save ensures gradients and model states are safely pulled out of XLA graphs before saving
            xm.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, save_path)
            
        if step >= flags['max_steps']:
            xm.master_print("🎉 Massive TPU Auto-Scaling Run Complete!")
            break

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. THE TPU LAUNCHER BLOCK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == '__main__':
    FLAGS = {
        'batch_size': 4,  # 4 * 8 cores = 32 effective batch size per step
        'lr': 1.5e-4,
        'max_steps': 15000 # 15k steps on TPU is equivalent to hundreds of hours on a Colab T4
    }
    
    # Preemptive Check: Does this machine only have 1 TPU core (like Colab v5e-1)?
    # We must check BEFORE calling xmp.spawn, because xmp.spawn can crash the C++ runtime on 1-core slices.
    try:
        devices = xm.get_xla_supported_devices()
    except Exception:
        devices = [1] # fallback assumption
        
    if len(devices) == 1:
        print("💡 Detected a Single-Core TPU environment (Colab v5e-1).")
        print("🚀 Bypassing multi-processing and executing 1.5B NEURON directly on the main thread...")
        _mp_fn(0, FLAGS)
    else:
        print(f"🚀 Spawning PyTorch XLA Spawner across {len(devices)} Google TRC TPU Cores...")
        try:
            xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=len(devices), start_method='fork')
        except Exception as e:
            print(f"\n⚠️ Spawner failed ({e}). Falling back to main thread...")
            _mp_fn(0, FLAGS)
