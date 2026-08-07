import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from src.dataset import UCF101Dataset
from src.hierarchical.stream_synthesizer import MultiStreamSynthesizer
from src.hierarchical.manager_policy import ManagerPolicy
from src.hierarchical.worker_agent import WorkerAgent

def train_hrl(epochs=30, k_streams=4):
    print("============================================================")
    print("Phase 2 H-RL Multi-Stream: Full Cloud GPU Training Loop")
    print("============================================================")
    
    # Automatically select GPU if available (which it will be on Colab T4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Accelerated Execution Device: {device.type.upper()}")
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    print("\n[1] Initializing UCF-101 Dataset...")
    dataset = UCF101Dataset(data_dir="data", split="train01", seq_len=150, allow_synthetic_fallback=True)
    print(f"Dataset Size: {len(dataset)} video sequences")
    
    print("\n[2] Initializing Manager & Worker Policies...")
    synthesizer = MultiStreamSynthesizer(dataset, K_streams=k_streams, max_sequence_length=150)
    manager = ManagerPolicy(K_streams=k_streams).to(device)
    worker = WorkerAgent(num_classes=101).to(device)
    
    manager_opt = optim.Adam(manager.parameters(), lr=1e-4)
    worker_opt = optim.Adam(worker.parameters(), lr=1e-4)
    
    h_worker = worker.init_hidden(batch_size=k_streams, device=device)
    prev_glimpses = [0] * k_streams
    
    # Calculate how many macro-steps approximate a full epoch
    # (Number of videos / Streams) * Avg Steps per Video
    MACRO_STEPS_PER_EPOCH = int((len(dataset) / k_streams) * 37)
    
    print(f"\n[3] Initiating Training for {epochs} Epochs ({MACRO_STEPS_PER_EPOCH} steps/epoch)...")
    
    for epoch in range(1, epochs + 1):
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
            
            budgets = manager.allocate_discrete_budgets(entropies, max_total_budget=20)
            
            # --- WORKER PHASE ---
            hop_actions = []
            step_loss = 0.0
            new_h_states = []
            
            for k in range(k_streams):
                frame_k = frames[k:k+1]
                h_k = h_worker[k:k+1].detach()
                glimpse_k = prev_glimpses[k]
                
                h_new, logits, q_values = worker.forward_step(frame_k, h_k, glimpse_k)
                new_h_states.append(h_new)
                
                # Cross-Entropy Classification Loss
                loss_cls = F.cross_entropy(logits, labels[k:k+1])
                step_loss += loss_cls
                
                # Q-learning Target
                action_idx = torch.argmax(q_values, dim=-1).item()
                hop, next_glimpse = worker.decode_action(action_idx)
                
                hop_actions.append(hop)
                prev_glimpses[k] = next_glimpse
                
            h_worker = torch.cat(new_h_states, dim=0)
            
            # Backpropagate
            step_loss.backward()
            worker_opt.step()
            manager_opt.step()
            
            total_loss += step_loss.item()
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
    train_hrl(epochs=30, k_streams=4)
