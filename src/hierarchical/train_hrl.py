import torch
import torch.optim as optim
from src.hierarchical.manager_policy import ManagerPolicy
from src.hierarchical.worker_agent import WorkerAgent
from src.hierarchical.stream_synthesizer import MultiStreamSynthesizer
import numpy as np

def train_hrl_loop(dataset, K=4, max_epochs=5):
    """
    Simulated Two-Stage H-RL Training Loop connecting the Manager and Worker.
    In practice, this loop runs with a PyTorch DataLoader mapped into the Synthesizer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing H-RL Architecture on {device}...")
    
    # 1. Initialize Modules
    synthesizer = MultiStreamSynthesizer(dataset, K_streams=K)
    manager = ManagerPolicy(K_streams=K).to(device)
    worker = WorkerAgent().to(device)
    
    # Optimizers
    manager_opt = optim.Adam(manager.parameters(), lr=1e-4)
    worker_opt = optim.Adam(worker.parameters(), lr=1e-4)
    
    # Worker hidden states for all K streams
    # Because streams are independent, we maintain K distinct hidden states (batch_size=K)
    h_worker = worker.init_hidden(batch_size=K, device=device)
    
    # Initialize previous glimpses (Start with Full Frame = 0)
    prev_glimpses = [0] * K
    
    global_step = 0
    
    for epoch in range(max_epochs):
        print(f"--- Epoch {epoch+1}/{max_epochs} ---")
        
        # In a real environment, we'd run for T macro-steps per epoch
        for macro_step in range(100): 
            # 1. Manager Phase: Observe Global Stream Entropies
            frames, labels, entropies = synthesizer.get_current_frames()
            frames = frames.to(device).float() # (K, C, H, W)
            entropies = entropies.to(device).unsqueeze(0) # (1, K)
            
            # Manager allocates budgets (e.g., max 20 total frames to process across all streams)
            budgets = manager.allocate_discrete_budgets(entropies, max_total_budget=20)
            budgets = budgets.squeeze(0).cpu().numpy() # (K,)
            
            # 2. Worker Phase: Execute Local Actions
            hop_actions = []
            
            for k in range(K):
                # Process stream k
                frame_k = frames[k:k+1] # (1, C, H, W)
                h_k = h_worker[k:k+1]
                glimpse_k = prev_glimpses[k]
                
                # Worker forward pass
                h_new, logits, q_values = worker.forward_step(frame_k, h_k, glimpse_k)
                h_worker[k:k+1] = h_new
                
                # Epsilon-greedy action selection for Worker
                # Selects from 24 joint actions
                action_idx = torch.argmax(q_values, dim=-1).item()
                hop, next_glimpse = worker.decode_action(action_idx)
                
                hop_actions.append(hop)
                prev_glimpses[k] = next_glimpse
                
                # (In full training: store transitions in Experience Replay, compute rewards, etc.)
                
            # 3. Environment Step: Advance CCTV streams by predicted hops
            synthesizer.step_streams(hop_actions)
            global_step += 1
            
        print(f"Completed {global_step} macro-steps. Simulated Manager-Worker pipeline robust.")
        
    print("H-RL Training Simulation Complete.")
    return manager, worker

if __name__ == "__main__":
    # Dummy dataset generator for quick local verification
    print("Generating Dummy Surveillance Dataset for Verification...")
    dummy_dataset = []
    for _ in range(50):
        video = torch.randn(100, 3, 112, 112) # T, C, H, W
        label = np.random.randint(0, 101)
        dummy_dataset.append((video, label))
        
    train_hrl_loop(dummy_dataset, K=4, max_epochs=2)
