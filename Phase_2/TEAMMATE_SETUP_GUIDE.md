# Phase 2: Local Teammate GPU Setup Guide

To ensure you don't spend 9+ hours running a single epoch on your CPU, follow these steps to properly configure your local environment to utilize your NVIDIA GPU!

## 1. Sync the Latest Code
Make sure you have pulled the latest optimizations (including the Windows CPU safety checks and VRAM limits).
```bash
git pull origin master
```

## 2. CRITICAL: Install CUDA-Enabled PyTorch
By default, `pip` on Windows often installs the CPU-only version of PyTorch. You MUST force it to install the CUDA 12.1 version so it can detect your NVIDIA GPU.
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 3. Install Remaining Dependencies
```bash
pip install opencv-python numpy psutil
```

## 4. Run the 1-Epoch Timing Test!
Run the training script but limit it to 1 epoch to ensure everything is working and to benchmark your machine's speed.
```bash
python Phase_2/scripts/train_full_hrl.py --epochs 1
```

---

### 🚨 How to Verify Your GPU is Working:
When you run Step 4, watch the first few lines of the terminal output closely. 

✅ **SUCCESS:**
If it says `Accelerated Execution Device: CUDA (1 GPUs detected)`, you are perfectly configured! A single epoch will likely take around 15-25 minutes.

❌ **FAILURE:**
If it says `Accelerated Execution Device: CPU (0 GPUs detected)`, **stop the script immediately (Ctrl+C).** This means PyTorch cannot see your GPU and is using your CPU. If you let it run, a single epoch will take over 9 hours. 
*(Fix: Update your NVIDIA graphics drivers using GeForce Experience, then re-run Step 2).*
