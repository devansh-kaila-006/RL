import os
import time
import torch
import argparse
from src.dataset import get_ucf_loaders
from src.model import AdaFrameModel
from src.agent import AdaFrameAgent
from src.evaluate import evaluate_models_comparative

def run_phase1_training(epochs=3, batch_size=4, seq_len=16, use_full_dataset=False, max_batches=None, run_name="full_run"):
    print("=" * 80)
    print(" ADAFRAME-RL MINI PROJECT: PHASE 1 MODEL TRAINING & EVALUATION WORKFLOW")
    print(" Team: Devansh Kaila (BL.EN.U4AIE23004) & Akshay Damera (BL.EN.U4AIE23072)")
    print(" Faculty Guide: Dr. Amudha J., Professor")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Operating Compute Device: {device}")
    
    # 1. Select prototyping active class subset vs full 101 class evaluation
    if use_full_dataset:
        subset_classes = None
        print("[Dataset] Mode: Full UCF-101 (101 Action Categories)")
    else:
        subset_classes = ["Basketball", "Pushups", "BenchPress", "SoccerJumping", "PlayingGuitar"]
        print(f"[Dataset] Mode: Fast Prototyping Subset ({len(subset_classes)} Action Categories: {', '.join(subset_classes)})")
        
    print("[Dataset] Initializing train and test video split data loaders...")
    train_loader, test_loader, idx_to_class = get_ucf_loaders(
        data_dir="data", 
        batch_size=batch_size, 
        seq_len=seq_len, 
        subset_classes=subset_classes, 
        allow_synthetic=True,
        num_workers=0
    )
    num_classes = len(idx_to_class)
    print(f"[Dataset] Initialized successfully with {num_classes} active prediction target classes.")

    # 2. Initialize AdaFrame-RL Model and Agent
    print("\n[Architecture] Instantiating AdaFrameModel (MobileNetV2 backbone + GRU + DQN Policy)...")
    model = AdaFrameModel(num_classes=num_classes, feature_dim=512, hidden_dim=256, frame_hops=[1, 2, 4, 8]).to(device)
    agent = AdaFrameAgent(model, lr=5e-4, gamma=0.95)
    print("[Architecture] Network weights initialized and transferred to target compute device.")

    # 3. Execution Training Loop
    print(f"\n[Training] Commencing multi-objective optimization across {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        total_batches = 0
        total_frames_inspected = 0.0
        total_hop_size = 0.0
        
        for i, (video_tensor, targets, _) in enumerate(train_loader):
            video_tensor = video_tensor.to(device)
            targets = targets.to(device)
            
            _, avg_inspected, avg_hop = agent.run_video_episode(video_tensor, targets, is_training=True)
            
            total_batches += 1
            total_frames_inspected += avg_inspected
            total_hop_size += avg_hop
            
            # Print periodic progress
            if (i + 1) % 5 == 0 or i == len(train_loader) - 1:
                print(f"  Epoch [{epoch}/{epochs}] Batch [{i+1}/{len(train_loader)}] | "
                      f"Avg Frames Viewed: {avg_inspected:.1f}/{seq_len} | "
                      f"Avg Hop Stride: {avg_hop:.1f}f | Epsilon: {agent.epsilon:.2f}")
                      
            if max_batches is not None and (i + 1) >= max_batches:
                print(f"  [Notice] Reached specified max_batches limit ({max_batches}). Concluding epoch.")
                break
                
        mean_inspected = total_frames_inspected / max(1, total_batches)
        mean_hop = total_hop_size / max(1, total_batches)
        print(f"-> Epoch [{epoch}/{epochs}] Summary: Avg Frame Inspection: {mean_inspected:.1f}/{seq_len} "
              f"({100.0 * (1.0 - mean_inspected/seq_len):.1f}% reduction) | Mean Action Stride: {mean_hop:.1f} frames")

    training_duration = time.time() - start_time
    print(f"\n[Training Complete] Total elapsed training time: {training_duration:.2f} seconds.")

    # 4. Save checkpoint
    checkpoint_path = os.path.join("data", f"adaframe_checkpoint_phase1_{run_name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'subset_classes': subset_classes,
        'frame_hops': model.frame_hops
    }, checkpoint_path)
    print(f"[Checkpoint] Saved trained model weights to: {checkpoint_path}")

    # 5. Comparative Evaluation against Naive Exhaustive Baseline
    evaluate_models_comparative(model, agent, test_loader, idx_to_class, device=device, max_batches=max_batches)
    print("Workflow executed successfully! You are ready to analyze findings and scale to Phase 2.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaFrame-RL Phase 1 Training and Benchmark Suite")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for video sequences")
    parser.add_argument("--seq-len", type=int, default=16, help="Temporal length of extracted video clips")
    parser.add_argument("--full-dataset", action="store_true", help="Evaluate across all 101 UCF classes instead of 5-class demo subset")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional ceiling on batches per epoch for prototyping verification")
    parser.add_argument("--run-name", type=str, default="full_run", help="Identifier name to isolate saved checkpoints and experiment results")
    args = parser.parse_args()
    
    run_phase1_training(epochs=args.epochs, batch_size=args.batch_size, seq_len=args.seq_len, use_full_dataset=args.full_dataset, max_batches=args.max_batches, run_name=args.run_name)
