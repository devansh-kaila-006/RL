import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import UCF101Dataset
from src.hierarchical.manager_policy import ManagerPolicy
from src.hierarchical.worker_agent import WorkerAgent

def evaluate_hrl(checkpoint_path, num_videos=100):
    print("============================================================")
    print("Phase 2 H-RL: Demo Evaluation & Benchmarking")
    print("============================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Accelerated Execution Device: {device.type.upper()}")
    print(f"Loading checkpoint {checkpoint_path}...")
    
    # Use the test01 split for proper scientific evaluation
    dataset = UCF101Dataset(data_dir="data", split="test01", seq_len=150, allow_synthetic_fallback=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Must match the K_streams used during training to successfully load the weights!
    manager = ManagerPolicy(K_streams=16).to(device)
    worker = WorkerAgent(num_classes=101).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DataParallel prefix if the model was saved on dual GPUs
    def clean_state_dict(sd):
        return {k.replace('module.', ''): v for k, v in sd.items()}
    
    manager.load_state_dict(clean_state_dict(checkpoint['manager_state_dict']))
    worker.load_state_dict(clean_state_dict(checkpoint['worker_state_dict']))
    
    manager.eval()
    worker.eval()
    
    correct = 0
    total = 0
    total_baseline_frames = 0
    total_processed_frames = 0
    
    start_time = time.time()
    num_to_evaluate = min(num_videos, len(dataset))
    print(f"\nEvaluating exactly {num_to_evaluate} videos from the Test Split...")
    
    with torch.no_grad():
        for i, (video, label, _) in enumerate(dataloader):
            if i >= num_videos:
                break
                
            video = video.to(device).float()
            label = label.to(device)
            T = video.size(1)
            
            # Real spatial entropy (Standard deviation of pixel intensities, matching training phase)
            entropy = torch.zeros(1, 16, device=device)
            # The initial frame is frame 0
            entropy[0, 0] = video[:, 0, :, :, :].float().std().item()
            
            # [1] MANAGER POLICY: Allocate processing budget for this specific video
            budget_ratio = manager(entropy)
            budget = (budget_ratio[0, 0] * 20).floor().long().clamp(min=1).item()
            
            h_worker = worker.init_hidden(batch_size=1, device=device)
            prev_glimpse = torch.tensor([0], dtype=torch.long, device=device)
            
            current_frame_idx = 0
            frames_processed = 0
            
            # [2] WORKER POLICY: Process the video using temporal hops and spatial glimpses
            while current_frame_idx < T and frames_processed < budget:
                frame = video[:, current_frame_idx, :, :, :]
                h_new, logits, q_values = worker(frame, h_worker, prev_glimpse)
                
                action_idx = torch.argmax(q_values, dim=-1).item()
                hop, next_glimpse = worker.decode_action(action_idx)
                
                h_worker = h_new
                prev_glimpse = torch.tensor([next_glimpse], dtype=torch.long, device=device)
                
                current_frame_idx += hop
                frames_processed += 1
                
            # Final Classification Prediction
            pred = torch.argmax(logits, dim=-1)
            if pred == label:
                correct += 1
            total += 1
            
            total_baseline_frames += T
            total_processed_frames += frames_processed
            
            if (i+1) % 10 == 0:
                acc = 100.0 * correct / total
                red = 100.0 * (1.0 - (total_processed_frames / total_baseline_frames))
                print(f"  Processed {i+1}/{num_to_evaluate} | Acc: {acc:.1f}% | Compute Reduction: {red:.1f}%")

    end_time = time.time()
    
    accuracy = 100.0 * correct / total
    compute_reduction = 100.0 * (1.0 - (total_processed_frames / total_baseline_frames))
    fps = total_processed_frames / (end_time - start_time)
    
    print("\n==================================================")
    print(f"H-RL EVALUATION RESULTS")
    print(f"Total Videos: {total}")
    print(f"Classification Accuracy: {accuracy:.2f}%")
    print(f"Total Baseline Frames: {total_baseline_frames}")
    print(f"Total Processed Frames: {total_processed_frames}")
    print(f"Compute Waste Reduction: {compute_reduction:.2f}%")
    print(f"Inference Speed: {fps:.2f} FPS")
    print("==================================================")
    
    # ---------------------------------------------------------
    # PLOTTING METRICS
    # ---------------------------------------------------------
    os.makedirs("Phase_2/results", exist_ok=True)
    
    # Plot 1: Compute Reduction Bar Chart
    plt.figure(figsize=(8, 6))
    sns.set_style("darkgrid")
    bars = plt.bar(
        ['Baseline (100% Frames)', 'AdaFrame H-RL (Active Vision)'], 
        [total_baseline_frames, total_processed_frames], 
        color=['#e74c3c', '#2ecc71']
    )
    plt.title(f"AdaFrame-RL Compute Reduction: {compute_reduction:.1f}% Savings", fontsize=14, pad=15)
    plt.ylabel("Total Frames Computed by GPU", fontsize=12)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (total_baseline_frames * 0.02), f'{int(yval):,}', va='bottom', ha='center', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    plot_path = "Phase_2/results/compute_reduction_chart.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved Scientific Graph to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate H-RL Model Checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth file")
    parser.add_argument("--num_videos", type=int, default=100, help="Number of test videos to evaluate")
    args = parser.parse_args()
    
    evaluate_hrl(args.checkpoint, args.num_videos)
