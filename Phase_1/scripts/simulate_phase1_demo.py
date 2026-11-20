"""
AdaFrame-RL Phase 1: Interactive Demonstration & Simulation Engine
==================================================================
Simulates active reinforcement learning frame selection on real UCF-101 video clips.
Visualizes step-by-step agent decisions, Q-value distributions, frame hopping,
class confidence evolution, and comparative compute savings vs. exhaustive baseline.

Authors: Devansh Kaila (BL.EN.U4AIE23004) & Akshay Damera (BL.EN.U4AIE23072)
Faculty Guide: Dr. Amudha J., Professor
"""

import os
import sys
import time
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root is in python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.model import AdaFrameModel
from src.dataset import UCF101Dataset

SUPPORTED_CLASSES = ["Basketball", "BenchPress", "PlayingGuitar", "PushUps", "SoccerJuggling"]

def load_simulation_assets(checkpoint_path="data/adaframe_checkpoint_verified_5class.pth"):
    device = torch.device("cpu")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt.get("num_classes", 101)
    frame_hops = ckpt.get("frame_hops", [1, 2, 4, 8])
    subset_classes = ckpt.get("subset_classes", SUPPORTED_CLASSES)
    
    model = AdaFrameModel(num_classes=num_classes, frame_hops=frame_hops).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    dataset = UCF101Dataset(
        data_dir="data", split="test01", seq_len=16, 
        subset_classes=subset_classes, allow_synthetic_fallback=True
    )
    return model, dataset, frame_hops, dataset.idx_to_class

def render_ascii_timeline(inspected_indices, total_frames=16, current_step=None):
    timeline = []
    for f in range(total_frames):
        if f in inspected_indices:
            idx = inspected_indices.index(f) + 1
            timeline.append(f"[\033[92m★F{f:02d}\033[0m]")
        else:
            timeline.append(f"[ \033[90mF{f:02d}\033[0m ]")
    return " ".join(timeline)

def render_q_bars(q_values, frame_hops, chosen_idx):
    lines = []
    q_min, q_max = min(q_values), max(q_values)
    span = max(1e-5, q_max - q_min)
    for i, (hop, q) in enumerate(zip(frame_hops, q_values)):
        norm = int(((q - q_min) / span) * 18) if span > 0 else 5
        bar = "█" * norm + "░" * (18 - norm)
        prefix = "👉 " if i == chosen_idx else "   "
        style = "\033[92m" if i == chosen_idx else "\033[90m"
        lines.append(f"{prefix}{style}Hop {hop:1d}f | {bar} Q={q:6.2f}\033[0m")
    return "\n".join(lines)

def run_simulation(model, dataset, idx_to_class, target_class=None, video_index=None, interactive=False, save_figure=True):
    print("\n" + "═" * 80)
    print("  ADAFRAME-RL: ACTIVE REINFORCEMENT LEARNING VIDEO INFERENCE SIMULATOR")
    print("  Phase 1 Architecture: MobileNetV2 + Recurrent GRU + Deep Q-Network Head")
    print("═" * 80)
    
    # Select video
    candidates = []
    for idx, (path_str, cls_idx) in enumerate(dataset.video_items):
        cname = dataset.idx_to_class.get(cls_idx, "")
        if target_class is None or cname.lower() == target_class.lower():
            candidates.append(idx)
            
    if not candidates:
        print(f"[Warning] No videos found for class '{target_class}'. Selecting from entire test split.")
        candidates = list(range(len(dataset)))
        
    chosen_idx = video_index if (video_index is not None and video_index < len(dataset)) else random.choice(candidates)
    video_tensor, label_idx, rel_path = dataset[chosen_idx]
    video_tensor = video_tensor.unsqueeze(0) # (1, T, C, H, W)
    true_class = idx_to_class.get(label_idx, f"Class_{label_idx}")
    
    print(f"\n[Test Video Clip Selected]")
    print(f"  • Source File:    data/UCF-101/{rel_path}")
    print(f"  • Ground Truth:   {true_class}")
    print(f"  • Total Duration: 16 Uniform Temporal Frames")
    print(f"  • Baseline Cost:  16 full convolutional evaluations (100% sequential)")
    print("-" * 80)
    
    # 1. Run Exhaustive Baseline for Ground Truth Reference
    t_start = time.perf_counter()
    with torch.no_grad():
        base_logits, base_eval_frames = model.forward_exhaustive_baseline(video_tensor)
    base_latency = (time.perf_counter() - t_start) * 1000.0
    base_probs = F.softmax(base_logits, dim=-1).squeeze(0)
    base_pred_idx = torch.argmax(base_probs).item()
    base_pred_class = idx_to_class.get(base_pred_idx, f"Class_{base_pred_idx}")
    base_pred_conf = base_probs[base_pred_idx].item() * 100.0
    
    # 2. Run AdaFrame-RL Simulation Step-by-Step
    h_t = model.init_hidden(1, device=torch.device("cpu"))
    frame_idx = 0
    step = 0
    max_steps = 10
    
    trajectory = []
    inspected_frames = []
    all_raw_frames = []
    
    # Un-normalize frames for plotting
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    for f in range(16):
        raw = video_tensor[0, f].permute(1, 2, 0).cpu().numpy()
        raw = np.clip((raw * std + mean) * 255.0, 0, 255).astype(np.uint8)
        all_raw_frames.append(raw)
        
    rl_t_start = time.perf_counter()
    
    print("\n[ACTIVE RL EPISODE COMMENCING]")
    while frame_idx < 16 and step < max_steps:
        inspected_frames.append(frame_idx)
        frame_tensor = video_tensor[:, frame_idx, :, :, :]
        
        with torch.no_grad():
            h_new, logits, q_vals = model.forward_frame_step(frame_tensor, h_t)
            
        q_list = q_vals.squeeze(0).tolist()
        action_idx = torch.argmax(q_vals, dim=-1).item()
        hop_selected = model.frame_hops[action_idx]
        
        probs = F.softmax(logits, dim=-1).squeeze(0)
        pred_idx = torch.argmax(probs).item()
        pred_class = idx_to_class.get(pred_idx, f"Class_{pred_idx}")
        pred_conf = probs[pred_idx].item() * 100.0
        
        step_info = {
            "step": step + 1,
            "frame_idx": frame_idx,
            "q_values": q_list,
            "action_idx": action_idx,
            "hop": hop_selected,
            "pred_class": pred_class,
            "pred_conf": pred_conf,
            "probs": probs.cpu().numpy()
        }
        trajectory.append(step_info)
        
        # Display Step Details
        print(f"\n┌── \033[1mSTEP {step+1}: Inspecting Frame F{frame_idx:02d} / 16\033[0m " + "─" * 45)
        print(f"│ Current Timeline:  {render_ascii_timeline(inspected_frames, 16)}")
        print(f"│ Predicted Class:   \033[93m{pred_class}\033[0m (Confidence: {pred_conf:.1f}%)")
        print(f"│ DQN Action Q-Values:")
        for q_line in render_q_bars(q_list, model.frame_hops, action_idx).split("\n"):
            print(f"│   {q_line}")
            
        skipped_str = f"F{frame_idx+1:02d}..F{min(15, frame_idx + hop_selected - 1):02d}" if hop_selected > 1 else "None"
        print(f"│ ⚡ Decision: Take \033[92mHop = {hop_selected} frames\033[0m ───> Skipping [{skipped_str}]")
        print("└" + "─" * 70)
        
        if interactive:
            try:
                input("Press [Enter] to step agent forward...")
            except (KeyboardInterrupt, EOFError):
                interactive = False
                
        h_t = h_new
        frame_idx += hop_selected
        step += 1
        
    rl_latency = (time.perf_counter() - rl_t_start) * 1000.0
    
    # Final Classifier Evaluation
    with torch.no_grad():
        final_logits = model.classifier(h_t)
        final_probs = F.softmax(final_logits, dim=-1).squeeze(0)
        final_pred_idx = torch.argmax(final_probs).item()
        final_pred_class = idx_to_class.get(final_pred_idx, f"Class_{final_pred_idx}")
        final_conf = final_probs[final_pred_idx].item() * 100.0
        
    num_inspected = len(inspected_frames)
    frame_reduction = (1.0 - (num_inspected / 16.0)) * 100.0
    speedup_ratio = 16.0 / max(1.0, num_inspected)
    is_correct = (final_pred_class == true_class)
    
    # 3. Print Executive Summary Comparison
    print("\n" + "═" * 80)
    print("                     SIMULATION BENCHMARK SUMMARY")
    print("═" * 80)
    print(f"{'Performance Metric':<30} │ {'Exhaustive Baseline':<20} │ {'AdaFrame-RL Agent':<20}")
    print("─" * 80)
    print(f"{'Classification Prediction':<30} │ {base_pred_class:<20} │ {final_pred_class:<20}")
    print(f"{'Confidence Level':<30} │ {base_pred_conf:>18.1f} % │ {final_conf:>18.1f} %")
    verdict_base = "CORRECT" if base_pred_class == true_class else "MISCLASSIFIED"
    verdict_rl = "CORRECT" if is_correct else "MISCLASSIFIED"
    print(f"{'Ground Truth Match':<30} │ {verdict_base:<20} │ {verdict_rl:<20}")
    print(f"{'Frames Inspected':<30} │ {16:>18d}   │ {num_inspected:>18d}  ")
    print(f"{'Inspected Frame Indices':<30} │ {'0 to 15 (All)':<20} │ {str(inspected_frames):<20}")
    print(f"{'Computational Frame Savings':<30} │ {'0.0 % (None)':>20} │ {frame_reduction:>18.1f} %")
    print(f"{'Estimated FLOP Speedup':<30} │ {'1.00x':>20} │ {speedup_ratio:>18.2f}x")
    print(f"{'Measured CPU Latency':<30} │ {base_latency:>17.1f} ms │ {rl_latency:>17.1f} ms")
    print("═" * 80)
    
    # 4. Generate Visual Figure if requested
    if save_figure:
        fig_path = generate_trajectory_figure(
            all_raw_frames, inspected_frames, trajectory, true_class, 
            final_pred_class, final_conf, num_inspected, frame_reduction, speedup_ratio, rel_path
        )
        print(f"\n[Artifact] High-Resolution Trajectory Figure Saved to:\n  -> {fig_path}")

    return {
        "true_class": true_class,
        "pred_class": final_pred_class,
        "is_correct": is_correct,
        "inspected_frames": inspected_frames,
        "num_inspected": num_inspected,
        "frame_reduction": frame_reduction,
        "speedup_ratio": speedup_ratio,
        "trajectory": trajectory
    }

def generate_trajectory_figure(frames, inspected, trajectory, true_class, pred_class, pred_conf, num_insp, reduction, speedup, rel_path):
    fig = plt.figure(figsize=(15, 9), facecolor="#0f172a")
    gs = fig.add_gridspec(3, 16, height_ratios=[1.8, 1.2, 1.0], hspace=0.45, wspace=0.15)
    
    # Row 1: All 16 Frames with Inspected vs Skipped Highlights
    for f in range(16):
        ax = fig.add_subplot(gs[0, f])
        ax.imshow(frames[f])
        ax.set_xticks([])
        ax.set_yticks([])
        
        if f in inspected:
            step_num = inspected.index(f) + 1
            for spine in ax.spines.values():
                spine.set_edgecolor("#22c55e") # Neon green
                spine.set_linewidth(3.5)
            ax.set_title(f"Step {step_num}\nF{f:02d}", color="#22c55e", fontsize=9, fontweight="bold", pad=4)
        else:
            ax.imshow(frames[f], alpha=0.30)
            for spine in ax.spines.values():
                spine.set_edgecolor("#475569") # Slate gray
                spine.set_linewidth(1.0)
                spine.set_linestyle("--")
            ax.set_title(f"F{f:02d}\n[SKIP]", color="#94a3b8", fontsize=8, pad=4)
            # Add an X marker over skipped frames
            ax.plot([0, 224], [0, 224], color="#ef4444", alpha=0.4, linewidth=1.5)
            ax.plot([0, 224], [224, 0], color="#ef4444", alpha=0.4, linewidth=1.5)

    # Row 2: Step-by-Step Q-Value Distributions
    num_steps = len(trajectory)
    cols_per_step = max(2, 16 // num_steps)
    
    for s_idx, step_data in enumerate(trajectory):
        start_c = s_idx * cols_per_step
        end_c = min(16, start_c + cols_per_step)
        ax_q = fig.add_subplot(gs[1, start_c:end_c])
        ax_q.set_facecolor("#1e293b")
        
        hops = [1, 2, 4, 8]
        q_vals = step_data["q_values"]
        chosen_hop = step_data["hop"]
        
        colors = ["#22c55e" if h == chosen_hop else "#64748b" for h in hops]
        bars = ax_q.bar([str(h) + "f" for h in hops], q_vals, color=colors, width=0.6, edgecolor="#334155")
        
        ax_q.set_title(f"Step {s_idx+1} (Frame {step_data['frame_idx']}) -> Hop {chosen_hop}f", color="#f8fafc", fontsize=9, fontweight="bold")
        ax_q.tick_params(colors="#cbd5e1", labelsize=8)
        ax_q.grid(axis="y", linestyle=":", alpha=0.3, color="#94a3b8")
        for spine in ax_q.spines.values():
            spine.set_edgecolor("#334155")

    # Row 3: Executive Metrics Dashboard Banner
    ax_card = fig.add_subplot(gs[2, :])
    ax_card.set_facecolor("#1e293b")
    ax_card.axis("off")
    for spine in ax_card.spines.values():
        spine.set_edgecolor("#38bdf8")
        spine.set_linewidth(1.5)
        
    title_text = f"ADAFRAME-RL PHASE 1 ACTIVE PERCEPTION SIMULATION: {true_class.upper()}"
    ax_card.text(0.5, 0.85, title_text, color="#38bdf8", fontsize=13, fontweight="bold", ha="center")
    
    metrics_str = (
        f"Ground Truth: {true_class}   │   Prediction: {pred_class} ({pred_conf:.1f}%)   │   "
        f"Frames Inspected: {num_insp}/16   │   Compute Reduction: {reduction:.1f}%   │   "
        f"FLOP Speedup: {speedup:.2f}x"
    )
    ax_card.text(0.5, 0.45, metrics_str, color="#f8fafc", fontsize=11, ha="center")
    
    subtext = f"Sample: {rel_path}  •  Green Border: Keyframes Processed  •  Red X: Redundant Frames Vaulted"
    ax_card.text(0.5, 0.15, subtext, color="#94a3b8", fontsize=9, ha="center")

    out_dir = os.path.join("Phase_1", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo_simulation_trajectory.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaFrame-RL Phase 1 Interactive Simulation")
    parser.add_argument("--class-name", type=str, default="Basketball", 
                        choices=SUPPORTED_CLASSES + ["any"], help="Action class to simulate")
    parser.add_argument("--video-idx", type=int, default=None, help="Explicit video index in test loader")
    parser.add_argument("--interactive", action="store_true", help="Pause on each step for interactive inspection")
    parser.add_argument("--no-figure", action="store_true", help="Skip saving trajectory PNG visualization")
    args = parser.parse_args()
    
    target_cls = None if args.class_name == "any" else args.class_name
    model, dataset, frame_hops, idx_to_class = load_simulation_assets()
    run_simulation(
        model, dataset, idx_to_class, 
        target_class=target_cls, 
        video_index=args.video_idx, 
        interactive=args.interactive,
        save_figure=not args.no_figure
    )
