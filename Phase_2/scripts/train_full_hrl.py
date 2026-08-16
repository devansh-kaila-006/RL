import os
import sys
import time
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import UCF101Dataset
from src.hierarchical.stream_synthesizer import MultiStreamSynthesizer
from src.hierarchical.manager_policy import ManagerPolicy
from src.hierarchical.worker_agent import WorkerAgent

def train_hrl(epochs=30, k_streams=16, resume_path=None):
    print("============================================================")
    print("Phase 2 H-RL Multi-Stream: Full Cloud GPU Training Loop")
    print("============================================================")
    
    # Automatically select GPU if available (which it will be on Colab T4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Accelerated Execution Device: {device.type.upper()} ({num_gpus} GPUs detected)")
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    print("\n[1] Initializing UCF-101 Dataset & Asynchronous Prefetch Queue...")
    dataset = UCF101Dataset(data_dir="data", split="train01", seq_len=150, allow_synthetic_fallback=True)
    print(f"Dataset Size: {len(dataset)} video sequences")
    
    # Asynchronous DataLoader (num_workers=0 on Windows to prevent freezing, 4 on Linux/Kaggle for speed)
    safe_workers = 4 if os.name != 'nt' else 0
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True, 
        num_workers=safe_workers, 
        prefetch_factor=2 if safe_workers > 0 else None, 
        pin_memory=True
    )
    
    def infinite_dataloader():
        while True:
            for batch in dataloader:
                yield batch
                
    data_iterator = iter(infinite_dataloader())
    
    print(f"\n[2] Initializing Manager & Worker Policies for {k_streams} Streams...")
    synthesizer = MultiStreamSynthesizer(data_iterator, K_streams=k_streams, max_sequence_length=150)
    
    manager = ManagerPolicy(K_streams=k_streams).to(device)
    worker = WorkerAgent(num_classes=101).to(device)
    
    # Dual-GPU / Multi-GPU Optimization!
    if num_gpus > 1:
        print(f"Optimizing workloads across {num_gpus} GPUs using DataParallel...")
        manager = torch.nn.DataParallel(manager)
        worker = torch.nn.DataParallel(worker)
    
    manager_opt = optim.Adam(manager.parameters(), lr=1e-4)
    worker_opt = optim.Adam(worker.parameters(), lr=1e-4)
    
    # The true worker agent instance for calling init_hidden and decode_action
    worker_base = worker.module if isinstance(worker, torch.nn.DataParallel) else worker
    
    start_epoch = 1
    if resume_path and os.path.exists(resume_path):
        print(f"\n[!] Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        manager.load_state_dict(checkpoint['manager_state_dict'])
        worker.load_state_dict(checkpoint['worker_state_dict'])
        manager_opt.load_state_dict(checkpoint['manager_opt_dict'])
        worker_opt.load_state_dict(checkpoint['worker_opt_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Successfully loaded weights! Resuming at Epoch {start_epoch}...")
        
    h_worker = worker_base.init_hidden(batch_size=k_streams, device=device)
    prev_glimpses = [0] * k_streams
    
    # Calculate how many macro-steps approximate a full epoch
    MACRO_STEPS_PER_EPOCH = int((len(dataset) / k_streams) * 37)
    
    print(f"\n[3] Initiating Training for {epochs} Epochs ({MACRO_STEPS_PER_EPOCH} steps/epoch)...")
    
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        start_time = time.time()
        
        manager.train()
        worker.train()
        total_loss = 0.0
        
        for step in range(1, MACRO_STEPS_PER_EPOCH + 1):
            manager_opt.zero_grad()
            worker_opt.zero_grad()
            
            # --- MANAGER PHASE ---
            frames, labels, entropies = synthesizer.get_current_frames()
            frames = frames.to(device).float()
            entropies = entropies.to(device).unsqueeze(0)
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            
            # If wrapped in DataParallel, we still pass entropies.
            budgets = manager(entropies)
            # Depending on if DataParallel split the batch, we handle discrete conversion
            budgets = (budgets * 20).floor().long()
            budgets = torch.clamp(budgets, min=1)
            
            # --- WORKER PHASE (VECTORIZED BATCH EXECUTION) ---
            # Instead of a Python for loop, we process all 32 streams simultaneously on the GPU!
            h_worker = h_worker.detach()
            glimpse_tensor = torch.tensor(prev_glimpses, dtype=torch.long, device=device)
            
            h_new, logits, q_values = worker(frames, h_worker, glimpse_tensor)
            h_worker = h_new
            
            # Cross-Entropy Classification Loss (Vectorized)
            loss_cls = F.cross_entropy(logits, labels)
            
            # Q-learning Target Decoding
            action_indices = torch.argmax(q_values, dim=-1)
            
            hop_actions = []
            new_glimpses = []
            
            # Decoding is lightweight enough for CPU loop, but network forward/backward was massively batched!
            for k in range(k_streams):
                hop, next_glimpse = worker_base.decode_action(action_indices[k].item())
                hop_actions.append(hop)
                new_glimpses.append(next_glimpse)
                
            prev_glimpses = new_glimpses
            
            # Backpropagate
            loss_cls.backward()
            worker_opt.step()
            manager_opt.step()
            
            total_loss += loss_cls.item()
            synthesizer.step_streams(hop_actions)
            
            if step % 250 == 0:
                avg_loss = total_loss / 250
                print(f"  Step [{step}/{MACRO_STEPS_PER_EPOCH}] | Avg Loss: {avg_loss:.4f}")
                total_loss = 0.0
                
        # Save Checkpoint after every epoch
        epoch_time = time.time() - start_time
        checkpoint_path = f"checkpoints/hrl_checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'manager_state_dict': manager.state_dict(),
            'worker_state_dict': worker.state_dict(),
            'manager_opt_dict': manager_opt.state_dict(),
            'worker_opt_dict': worker_opt.state_dict(),
        }, checkpoint_path)
        print(f"Epoch {epoch} Completed in {epoch_time:.2f}s! Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 H-RL Multi-Stream GPU Training")
    parser.add_argument("--epochs", type=int, default=30, help="Total epochs to train")
    parser.add_argument("--k_streams", type=int, default=16, help="Number of concurrent streams (16 for laptop GPU, 32 for Kaggle)")
    parser.add_argument("--resume", type=str, default=None, help="Path to .pth checkpoint to resume from")
    args = parser.parse_args()
    
    train_hrl(epochs=args.epochs, k_streams=args.k_streams, resume_path=args.resume)
