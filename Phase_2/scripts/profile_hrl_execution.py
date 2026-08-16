import os
import sys
import time
import torch
import psutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import UCF101Dataset
from src.hierarchical.stream_synthesizer import MultiStreamSynthesizer
from src.hierarchical.manager_policy import ManagerPolicy
from src.hierarchical.worker_agent import WorkerAgent

def run_profiler():
    print("="*60)
    print("Phase 2 H-RL Multi-Stream Real-Time Profiler")
    print("="*60)
    
    # Configuration
    K_STREAMS = 4
    MACRO_STEPS = 100
    
    device = torch.device("cpu")
    print(f"Hardware Constraint: Forced execution on {device.type.upper()} for Edge viability test.")
    
    print("\n1. Initializing Dataset and H-RL Modules...")
    
    # Load dataset (using 10 items for quick synthesis)
    dataset = UCF101Dataset(data_dir="data", split="train01", seq_len=150, allow_synthetic_fallback=True)
    # Cache a small dataset in memory to avoid I/O bottlenecks during profiling
    print("Pre-fetching memory-cached surveillance clips to isolate Neural Network timing...")
    cached_dataset = []
    for i in range(10):
        video, label, _ = dataset[i]
        cached_dataset.append((video, label))
        
    synthesizer = MultiStreamSynthesizer(cached_dataset, K_streams=K_STREAMS, max_sequence_length=150)
    manager = ManagerPolicy(K_streams=K_STREAMS).to(device)
    worker = WorkerAgent(num_classes=101).to(device)
    
    # Disable gradients for pure inference speed profiling
    manager.eval()
    worker.eval()
    
    h_worker = worker.init_hidden(batch_size=K_STREAMS, device=device)
    prev_glimpses = [0] * K_STREAMS
    
    print("\n2. Initiating Real-Time Execution Loop...")
    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss / (1024 * 1024)
    
    total_frames_processed = 0
    total_frames_vaulted = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for step in range(MACRO_STEPS):
            # Manager phase
            frames, labels, entropies = synthesizer.get_current_frames()
            frames = frames.to(device).float()
            entropies = entropies.to(device).unsqueeze(0)
            
            # Manager assigns budgets (just executing forward pass for latency)
            budgets = manager.allocate_discrete_budgets(entropies, max_total_budget=20)
            budgets = budgets.squeeze(0).cpu().numpy()
            
            hop_actions = []
            
            # Worker phase
            for k in range(K_STREAMS):
                frame_k = frames[k:k+1]
                h_k = h_worker[k:k+1]
                glimpse_k = prev_glimpses[k]
                
                h_new, logits, q_values = worker.forward_step(frame_k, h_k, glimpse_k)
                h_worker[k:k+1] = h_new
                
                action_idx = torch.argmax(q_values, dim=-1).item()
                hop, next_glimpse = worker.decode_action(action_idx)
                
                hop_actions.append(hop)
                prev_glimpses[k] = next_glimpse
                
                total_frames_processed += 1
                total_frames_vaulted += (hop - 1)
                
            synthesizer.step_streams(hop_actions)
            
    end_time = time.time()
    mem_end = process.memory_info().rss / (1024 * 1024)
    
    elapsed = end_time - start_time
    # Effective FPS includes the frames we vaulted over, showing the speedup of the algorithm!
    actual_fps = total_frames_processed / elapsed
    effective_fps = (total_frames_processed + total_frames_vaulted) / elapsed
    compute_reduction = (total_frames_vaulted / (total_frames_processed + total_frames_vaulted)) * 100
    
    print("\n" + "="*60)
    print("PROFILING RESULTS (CPU ONLY)")
    print("="*60)
    print(f"Total Wall Clock Time:      {elapsed:.3f} seconds")
    print(f"Total Macro-Steps:          {MACRO_STEPS} loops")
    print(f"Concurrent Camera Streams:  {K_STREAMS}")
    print(f"Active Frames Processed:    {total_frames_processed}")
    print(f"Redundant Frames Vaulted:   {total_frames_vaulted}")
    print("-" * 60)
    print(f"System Actual FPS:          {actual_fps:.2f} FPS (NN pass limit)")
    print(f"Effective Surveillance FPS: {effective_fps:.2f} FPS (Thanks to RL Vaulting!)")
    print(f"End-to-End Compute Savings: {compute_reduction:.1f}% frames eliminated")
    print("-" * 60)
    print(f"RAM Memory Footprint Delta: +{mem_end - mem_start:.2f} MB")
    print(f"Mean Latency per stream:    {(elapsed / (MACRO_STEPS * K_STREAMS) * 1000):.2f} ms")
    print("="*60)

if __name__ == "__main__":
    run_profiler()
