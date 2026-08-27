import os
import sys
import time
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F

# Insert Python path bootstrap (Fixes Issue 5)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import UCF101Dataset
from src.hierarchical.stream_synthesizer import MultiStreamSynthesizer
from src.hierarchical.manager_policy import ManagerPolicy
from src.hierarchical.worker_agent import WorkerAgent, ReplayBuffer

def train_hrl(epochs=30, k_streams=16, resume_path=None):
    print("============================================================")
    print("Phase 2 H-RL Multi-Stream: Full Cloud GPU Training Loop (RL FIXED)")
    print("============================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Accelerated Execution Device: {device.type.upper()} ({num_gpus} GPUs detected)")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    print("\n[1] Initializing UCF-101 Dataset & Asynchronous Prefetch Queue...")
    dataset = UCF101Dataset(data_dir="data", split="train01", seq_len=150, allow_synthetic_fallback=True)
    
    safe_workers = 4 if os.name != 'nt' else 0
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, 
        num_workers=safe_workers, prefetch_factor=2 if safe_workers > 0 else None, pin_memory=True
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
    replay_buffer = ReplayBuffer(capacity=50000)
    
    manager_opt = optim.Adam(manager.parameters(), lr=1e-4)
    worker_opt = optim.Adam(worker.parameters(), lr=1e-4)
    
    start_epoch = 1
    if resume_path and os.path.exists(resume_path):
        print(f"\n[!] Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        manager.load_state_dict(checkpoint['manager_state_dict'])
        worker.load_state_dict(checkpoint['worker_state_dict'])
        manager_opt.load_state_dict(checkpoint['manager_opt_dict'])
        worker_opt.load_state_dict(checkpoint['worker_opt_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
    # We maintain states across the epoch
    h_worker = worker.init_hidden(batch_size=k_streams, device=device)
    prev_glimpses = torch.zeros(k_streams, dtype=torch.long, device=device)
    
    MACRO_STEPS_PER_EPOCH = int((len(dataset) / k_streams) * 37)
    
    print(f"\n[3] Initiating Training for {epochs} Epochs ({MACRO_STEPS_PER_EPOCH} steps/epoch)...")
    
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        start_time = time.time()
        
        manager.train()
        worker.train()
        total_loss = 0.0
        
        for step in range(1, MACRO_STEPS_PER_EPOCH + 1):
            # --- MANAGER PHASE (REINFORCE) ---
            frames, labels, entropies = synthesizer.get_current_frames()
            frames = frames.to(device).float()
            entropies = entropies.to(device).unsqueeze(0)
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            
            # Predict discrete budgets (1-20) and save log_probs for backprop
            budgets, log_probs = manager.allocate_discrete_budgets(entropies)
            budgets = budgets.squeeze(0) # (K,)
            log_probs = log_probs.squeeze(0) # (K,)
            
            # --- WORKER PHASE (DQN & Cross-Entropy) ---
            # We process streams up to their allocated budget
            max_b = budgets.max().item()
            active_mask = torch.ones(k_streams, dtype=torch.bool, device=device)
            frames_processed = torch.zeros(k_streams, dtype=torch.long, device=device)
            
            manager_opt.zero_grad()
            worker_opt.zero_grad()
            
            # Track final logits for Manager Reward
            final_logits = torch.zeros(k_streams, 101, device=device)
            
            total_step_loss = 0.0
            
            for b_step in range(max_b):
                active_mask = frames_processed < budgets
                if not active_mask.any():
                    break
                    
                # We only process streams that still have budget. For simplicity in batching, 
                # we pass all streams but only update/push active ones.
                h_new, logits, q_values = worker(frames, h_worker, prev_glimpses)
                
                loss_cls = F.cross_entropy(logits[active_mask], labels[active_mask])
                total_step_loss = total_step_loss + loss_cls
                
                action_indices = torch.argmax(q_values, dim=-1)
                
                hop_actions = []
                new_glimpses = []
                
                for k in range(k_streams):
                    if active_mask[k]:
                        hop, next_glimpse = worker.decode_action(action_indices[k].item())
                        
                        # Calculate a step reward (1 if correct classification, else 0)
                        pred = torch.argmax(logits[k], dim=-1)
                        step_reward = 1.0 if pred == labels[k] else 0.0
                        
                        # Push to Experience Replay
                        replay_buffer.push(
                            h_worker[k], action_indices[k].item(), step_reward, h_new[k], done=False
                        )
                        
                        frames_processed[k] += 1
                        final_logits[k] = logits[k]
                    else:
                        hop, next_glimpse = 0, prev_glimpses[k].item()
                        
                    hop_actions.append(hop)
                    new_glimpses.append(next_glimpse)
                    
                # Update streams in synthesizer
                synthesizer.step_streams(hop_actions)
                frames, labels, _ = synthesizer.get_current_frames()
                frames = frames.to(device).float()
                labels = torch.tensor(labels, dtype=torch.long, device=device)
                
                # Update hidden state and glimpses for next step
                h_worker = torch.where(active_mask.unsqueeze(1), h_new, h_worker)
                prev_glimpses = torch.tensor(new_glimpses, dtype=torch.long, device=device)
                
                total_loss += loss_cls.item()
                
            # Backpropagate total step loss
            if isinstance(total_step_loss, torch.Tensor):
                total_step_loss.backward(retain_graph=True)
                
            # DQN Training from Replay Buffer
            if len(replay_buffer) > 256:
                state, action, reward, next_state, done = replay_buffer.sample(64)
                state, action, reward = state.to(device), action.to(device), reward.to(device)
                next_state, done = next_state.to(device), done.to(device)
                
                # Q(s, a)
                q_s = worker.policy_net(state)
                q_s_a = q_s.gather(1, action.unsqueeze(1)).squeeze(1)
                
                # Target = R + gamma * max Q(s')
                with torch.no_grad():
                    q_next = worker.policy_net(next_state).max(1)[0]
                    target_q = reward + 0.99 * q_next * (1 - done)
                    
                loss_q = F.mse_loss(q_s_a, target_q)
                loss_q.backward(retain_graph=True) # DQN gradients
                
            # --- MANAGER REWARD (Policy Gradient) ---
            # Reward = (Accuracy) - penalty * (Budget Used)
            final_preds = torch.argmax(final_logits, dim=-1)
            accuracy_reward = (final_preds == labels).float()
            
            # Heavy penalty for wasting frames
            efficiency_penalty = 0.05 * (budgets.float() / 20.0) 
            manager_reward = accuracy_reward - efficiency_penalty
            
            # Loss = -log_prob * Reward
            loss_manager = -(log_probs * manager_reward.detach()).mean()
            loss_manager.backward()
            
            manager_opt.step()
            worker_opt.step()
            
            h_worker = h_worker.detach()
            
            if step % 250 == 0:
                avg_loss = total_loss / (250 * max_b)
                print(f"  Step [{step}/{MACRO_STEPS_PER_EPOCH}] | Avg Cls Loss: {avg_loss:.4f} | Manager Reward: {manager_reward.mean().item():.3f}")
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
    parser.add_argument("--k_streams", type=int, default=16, help="Number of concurrent streams")
    parser.add_argument("--resume", type=str, default=None, help="Path to .pth checkpoint to resume from")
    args = parser.parse_args()
    
    train_hrl(epochs=args.epochs, k_streams=args.k_streams, resume_path=args.resume)
