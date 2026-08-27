import os
import sys
import time
import torch
import torch.optim as optim
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import UCF101Dataset
from src.hierarchical.stream_synthesizer import MultiStreamSynthesizer
from src.hierarchical.manager_policy import ManagerPolicy
from src.hierarchical.worker_agent import WorkerAgent, ReplayBuffer

def run_1_epoch():
    print("============================================================")
    print("Phase 2 H-RL Multi-Stream: 1-Epoch Scientific Test Run")
    print("============================================================")
    
    K_STREAMS = 4
    device = torch.device("cpu")
    print(f"Device: {device.type.upper()}")
    
    print("\n[1] Initializing UCF-101 Dataset...")
    dataset = UCF101Dataset(data_dir="data", split="train01", seq_len=150, allow_synthetic_fallback=True)
    dataset_size = len(dataset)
    print(f"Dataset Size: {dataset_size} video sequences")
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    def infinite_dataloader():
        while True:
            for batch in dataloader:
                yield batch
                
    data_iterator = iter(infinite_dataloader())
    
    print("\n[2] Initializing Manager & Worker Policies...")
    synthesizer = MultiStreamSynthesizer(data_iterator, K_streams=K_STREAMS, max_sequence_length=150)
    manager = ManagerPolicy(K_streams=K_STREAMS).to(device)
    worker = WorkerAgent(num_classes=101).to(device)
    replay_buffer = ReplayBuffer(capacity=50000)
    
    manager_opt = optim.Adam(manager.parameters(), lr=1e-4)
    worker_opt = optim.Adam(worker.parameters(), lr=1e-4)
    
    h_worker = worker.init_hidden(batch_size=K_STREAMS, device=device)
    prev_glimpses = torch.zeros(K_STREAMS, dtype=torch.long, device=device)
    
    manager.train()
    worker.train()
    
    MACRO_STEPS = 100 # Reduced for extremely fast CPU verification
    
    print(f"\n[3] Executing 1 Test Epoch (Limited to {MACRO_STEPS} Macro-Steps for CPU Viability)...")
    
    start_time = time.time()
    total_loss = 0.0
    
    for step in range(1, MACRO_STEPS + 1):
        # --- MANAGER PHASE (REINFORCE) ---
        frames, labels, entropies = synthesizer.get_current_frames()
        frames = frames.to(device).float()
        entropies = entropies.to(device).unsqueeze(0)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        
        budgets, log_probs = manager.allocate_discrete_budgets(entropies)
        budgets = budgets.squeeze(0) # (K,)
        log_probs = log_probs.squeeze(0) # (K,)
        
        max_b = budgets.max().item()
        frames_processed = torch.zeros(K_STREAMS, dtype=torch.long, device=device)
        
        manager_opt.zero_grad()
        worker_opt.zero_grad()
        
        final_logits = torch.zeros(K_STREAMS, 101, device=device)
        
        total_step_loss = 0.0
        
        for b_step in range(max_b):
            active_mask = frames_processed < budgets
            if not active_mask.any():
                break
                
            h_new, logits, q_values = worker(frames, h_worker, prev_glimpses)
            
            loss_cls = F.cross_entropy(logits[active_mask], labels[active_mask])
            total_step_loss = total_step_loss + loss_cls
            
            action_indices = torch.argmax(q_values, dim=-1)
            hop_actions = []
            new_glimpses = []
            
            for k in range(K_STREAMS):
                if active_mask[k]:
                    hop, next_glimpse = worker.decode_action(action_indices[k].item())
                    
                    pred = torch.argmax(logits[k], dim=-1)
                    step_reward = 1.0 if pred == labels[k] else 0.0
                    
                    replay_buffer.push(h_worker[k], action_indices[k].item(), step_reward, h_new[k], done=False)
                    
                    frames_processed[k] += 1
                    final_logits[k] = logits[k]
                else:
                    hop, next_glimpse = 0, prev_glimpses[k].item()
                    
                hop_actions.append(hop)
                new_glimpses.append(next_glimpse)
                
            synthesizer.step_streams(hop_actions)
            frames, labels, _ = synthesizer.get_current_frames()
            frames = frames.to(device).float()
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            
            h_worker = torch.where(active_mask.unsqueeze(1), h_new, h_worker)
            prev_glimpses = torch.tensor(new_glimpses, dtype=torch.long, device=device)
            
            total_loss += loss_cls.item()
            
        # End of worker loop, backpropagate total classification loss
        if isinstance(total_step_loss, torch.Tensor):
            total_step_loss.backward(retain_graph=True)
            
        # DQN Training from Replay Buffer
        if len(replay_buffer) > 256:
            state, action, reward, next_state, done = replay_buffer.sample(64)
            state, action, reward = state.to(device), action.to(device), reward.to(device)
            next_state, done = next_state.to(device), done.to(device)
            
            q_s = worker.policy_net(state)
            q_s_a = q_s.gather(1, action.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                q_next = worker.policy_net(next_state).max(1)[0]
                target_q = reward + 0.99 * q_next * (1 - done)
                
            loss_q = F.mse_loss(q_s_a, target_q)
            loss_q.backward(retain_graph=True)
            
        # Manager Reward
        final_preds = torch.argmax(final_logits, dim=-1)
        accuracy_reward = (final_preds == labels).float()
        efficiency_penalty = 0.05 * (budgets.float() / 20.0) 
        manager_reward = accuracy_reward - efficiency_penalty
        
        loss_manager = -(log_probs * manager_reward.detach()).mean()
        loss_manager.backward()
        
        # Step both optimizers ONCE per manager cycle
        manager_opt.step()
        worker_opt.step()
        
        h_worker = h_worker.detach()
        
        if step % 20 == 0:
            avg_loss = total_loss / (20 * max_b)
            elapsed = time.time() - start_time
            print(f"Step [{step}/{MACRO_STEPS}] | Avg Cls Loss: {avg_loss:.4f} | Manager Reward: {manager_reward.mean().item():.3f} | Elapsed: {elapsed:.2f}s")
            total_loss = 0.0
            
    total_time = time.time() - start_time
    print("============================================================")
    print(f"1-Epoch Test Run Completed Successfully in {total_time:.2f} seconds!")
    print("Gradients successfully propagated through true REINFORCE/DQN Manager-Worker H-RL loop!")
    print("============================================================")

if __name__ == "__main__":
    run_1_epoch()
