import time
import torch
import numpy as np

def evaluate_models_comparative(model, agent, test_loader, idx_to_class, device, max_batches=None):
    """
    Rigorous comparative evaluation adhering to ML best practices.
    Compares Naive Exhaustive Video Processing vs. AdaFrame-RL Dynamic Frame Hopping on fixed test data splits.
    
    Metrics Reported:
    - Classification Accuracy (%)
    - Average Frames Evaluated per sequence
    - Frame Reduction Ratio (%)
    - Inference Latency / Speedup Ratio (FLOP savings)
    """
    model.eval()
    total_samples = 0
    
    # Baseline stats
    base_correct = 0
    base_frames = 0
    base_time = 0.0
    
    # AdaFrame RL stats
    rl_correct = 0
    rl_frames = 0.0
    rl_time = 0.0
    
    # Confusion Matrix tracking for slice-based evaluation
    num_classes = len(idx_to_class)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    print("\n" + "="*75)
    print("COMPARATIVE MODEL EVALUATION: EXHAUSTIVE BASELINE vs ADAFRAME-RL AGENT")
    print("="*75)
    print(f"Evaluating across fixed test validation split on device: {device}...")
    
    with torch.no_grad():
        for i, (video_tensor, target, _) in enumerate(test_loader):
            video_tensor = video_tensor.to(device)
            target = target.to(device)
            batch_size = video_tensor.size(0)
            total_samples += batch_size
            
            # 1. Evaluate Exhaustive Baseline
            start_t = time.perf_counter()
            base_logits, eval_frames = model.forward_exhaustive_baseline(video_tensor)
            base_time += (time.perf_counter() - start_t)
            
            base_preds = torch.argmax(base_logits, dim=1)
            base_correct += (base_preds == target).sum().item()
            base_frames += (eval_frames * batch_size)
            
            # 2. Evaluate AdaFrame-RL Dynamic Hopping Agent
            start_t = time.perf_counter()
            rl_logits, avg_inspected, _ = agent.run_video_episode(video_tensor, target, is_training=False)
            rl_time += (time.perf_counter() - start_t)
            
            rl_preds = torch.argmax(rl_logits, dim=1)
            rl_correct += (rl_preds == target).sum().item()
            rl_frames += (avg_inspected * batch_size)
            
            # Populate confusion matrix for RL agent predictions
            for b in range(batch_size):
                t_idx = target[b].item()
                p_idx = rl_preds[b].item()
                if t_idx < num_classes and p_idx < num_classes:
                    confusion_matrix[t_idx, p_idx] += 1

            if (i + 1) % 5 == 0 or i == len(test_loader) - 1:
                print(f"[{i+1}/{len(test_loader)}] Batches evaluated...")
            if max_batches is not None and (i + 1) >= max_batches:
                break

    if total_samples == 0:
        print("No samples found in test split.")
        return

    # Calculate final comparative metrics
    base_acc = 100.0 * base_correct / total_samples
    rl_acc = 100.0 * rl_correct / total_samples
    
    avg_base_frames = base_frames / total_samples
    avg_rl_frames = rl_frames / total_samples
    frame_reduction = 100.0 * (1.0 - (avg_rl_frames / max(1.0, avg_base_frames)))
    speedup_ratio = max(1.0, avg_base_frames / max(1.0, avg_rl_frames))
    
    print("\n" + "-"*75)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("-" * 75)
    print(f"{'Metric':<32} | {'Exhaustive Baseline':<20} | {'AdaFrame-RL Agent':<20}")
    print("-" * 75)
    print(f"{'Classification Accuracy':<32} | {base_acc:>18.2f} % | {rl_acc:>18.2f} %")
    print(f"{'Avg Frames Processed / Clip':<32} | {avg_base_frames:>18.1f}   | {avg_rl_frames:>18.1f}  ")
    print(f"{'Frame Computational Reduction':<32} | {'0.00 % (None)':>20} | {frame_reduction:>18.2f} %")
    print(f"{'Estimated FLOP Speedup Factor':<32} | {'1.00x (Baseline)':>20} | {speedup_ratio:>18.2f}x")
    print("-" * 75)
    
    print("\n[ML Slice Analysis] AdaFrame-RL Agent Confusion Matrix across Active Subset:")
    print("-" * 65)
    col_width = 14
    header = f"{'Target \\ Pred':<16} | " + " ".join([f"{idx_to_class.get(i, f'C{i}')[:12]:<{col_width}}" for i in range(min(num_classes, 6))])
    print(header)
    print("-" * len(header))
    for r in range(min(num_classes, 6)):
        row_str = f"{idx_to_class.get(r, f'C{r}')[:14]:<16} | "
        for c in range(min(num_classes, 6)):
            row_str += f"{confusion_matrix[r, c]:<{col_width}} "
        print(row_str)
    print("=" * 75 + "\n")
    return base_acc, rl_acc, frame_reduction
