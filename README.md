# AdaFrame-RL: Adaptive Active Video Frame Selection via Reinforcement Learning
**Academic Project Repository & Empirical Research Suite**  
**Team Members:** Devansh Kaila (BL.EN.U4AIE23004) | Akshay Damera (BL.EN.U4AIE23072)  
**Faculty Guide:** Dr. Amudha J., Professor  

---

## 1. Project & Architecture Overview
In standard human action recognition paradigms, deep neural networks exhaustively process 100% of chronological video frames. This sequential approach introduces immense computational redundancy (FLOP wastage on static background scenes and repetitive motion), demanding high-end GPU hardware accelerators.

**AdaFrame-RL** completely overcomes this bottleneck by presenting an active visual perception framework that learns to dynamically hop across video frames in real time. Powered by Deep Q-Learning (DQN) coupled with a lightweight spatial-temporal recurrent network, the agent inspects informative keyframes while bounding past low-signal background segments.

```
+-------------------+      +-----------------------+      +--------------------------+
|  Input Video Clip | ---> | MobileNetV2 Extractor | ---> | Recurrent GRU Memory h_t |
+-------------------+      +-----------------------+      +-------------+------------+
                                                                        |
                                               +------------------------+------------------------+
                                               |                                                 |
                                               v                                                 v
                                  +------------------------+                        +------------------------+
                                  | Classification Logits  |                        |  DQN Policy Network    |
                                  |  (Action Prediction)   |                        | Hop A_t in {1, 2, 4, 8}|
                                  +------------------------+                        +------------------------+
```

---

## 2. Installation & Environment Setup
This repository is optimized to execute entirely on **standard CPU hardware** without requiring proprietary GPU dependencies or CUDA installations.

### Prerequisites & Dependency Locking
We enforce standard Python Virtual Environment runtime isolation (`venv + pip + requirements.txt`):

```powershell
# 1. Clone or navigate into the workspace repository
cd path\to\RL

# 2. Initialize a clean virtual environment (.venv)
python -m venv .venv

# 3. Activate virtual environment and install locked dependencies
.\.venv\Scripts\pip.exe install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

---

## 3. Usage Guide & Quick Start

### A. Training & Benchmarking from Scratch
To execute the unified training routine across official UCF-101 action subsets and monitor natural $\epsilon$-greedy exploration decay and memory replay convergence:
```powershell
.\.venv\Scripts\python.exe main_train.py
```
*Outputs real-time loss convergence, validation classification metrics, and preserves model weights to `data/`.*

### B. Regenerating Visual Analytics & Evaluation Plots
All plotting and empirical verification scripts reside in our dedicated test archive:
```powershell
.\.venv\Scripts\python.exe Phase_1\scripts\generate_all_visualizations_and_metrics.py
```

---

## 4. Repository Directory Structure
```
RL/
├── README.md                          # Primary project documentation & setup guide
├── requirements.txt                   # LOCKED UTF-8 dependency specifications
├── main_train.py                      # Primary model optimization & evaluation script
├── PROJECT_REFERENCE.md               # Historical academic specification baseline
│
├── src/                               # Core AI Architecture Modules
│   ├── __init__.py                    # Python package initialization
│   ├── model.py                       # MobileNetV2 Feature Backbone + GRU + DQN Head
│   ├── agent.py                       # Replay Memory Buffer + Epsilon-Greedy Policy Agent
│   ├── dataset.py                     # UCF-101 Video Loader & OpenCV Stream Decoder
│   └── evaluate.py                    # Comparative Benchmarking Engine (FLOPs/Latency)
│
├── data/                              # Model Checkpoints & Split Annotations
│   ├── adaframe_checkpoint_*.pth      # Trained & verified PyTorch parameter weights
│   └── splits/                        # Official UCF-101 Train/Test taxonomy mappings
│
├── EDA/                               # Extensive Exploratory Data Analysis Suite
│   ├── EDA_REPORT.md                  # Comprehensive statistical analysis & hypothesis testing
│   ├── run_dataset_eda.py             # Native OpenCV/NumPy analytical scanning pipeline
│   └── figures/                       # High-resolution dataset distribution and entropy charts
│
└── Phase_1/                           # Phase 1 Comprehensive Documentation & Figures
    ├── DEMO_RUN.md                    # Initial 200-sample prototyping demonstration results
    ├── FULL_RUN.md                    # Converged 5-epoch scientific run & 8-figure dashboard
    ├── ABLATIONS.md                   # Definitive ablation study, heatmaps, hardware benchmarks & failure cases
    ├── figures/                       # High-resolution PNG visual analytics library
    └── scripts/                       # Experimental validation & report generation scripts
```

---

## 5. Phase 1 Experimental Documentation & Configs
Our research findings are partitioned cleanly across three peer-reviewed analytical manuscripts inside the `Phase_1/` directory:

1. **[DEMO_RUN.md](Phase_1/DEMO_RUN.md):** Details our early prototyping evaluation across initial demonstration splits to confirm computational feasibility.
2. **[FULL_RUN.md](Phase_1/FULL_RUN.md):** Presents our complete unconstrained 5-epoch training run, documenting convergence curves, trimodal hopping equilibrium, and comparative efficiency against sequential baselines.
3. **[ABLATIONS.md](Phase_1/ABLATIONS.md):** Our comprehensive diagnostic paper featuring:
   * **5×5 Semantic Confusion Matrix Heatmap & Precision/Recall F1 Tables**
   * **Real-Time CPU Hardware Benchmarks** (FLOP reduction, inference runtime in ms, and RAM compression)
   * **Kinematic Saliency Discovery:** Proving why outdoor global action domains (*Basketball*, *SoccerJuggling*) empower maximal 8-frame bounds, while indoor sedentary postures (*BenchPress*, *PushUps*, *Guitar*) trigger shorter hop strides (~3–4 frames).
   * **Hyperparameter Sensitivity Grid:** 15-batch optimization loops verifying policy stability across learning rates and discount horizons.
   * **10-Class Scalability Benchmark & Qualitative Failure Analysis (Appendix A)**.

### Master Hyperparameter Configuration Profile
| Parameter / Architectural Variable | Value / Setting | Operational Role & Empirical Validation |
| :--- | :---: | :--- |
| **Spatial Backbone** | MobileNetV2 (Pre-trained) | Extractor weights frozen during early loops; fine-tuned with lower learning rate gradient updates. |
| **Temporal Memory Architecture** | GRU Cell (Hidden Dim: 256) | Preserves continuous sequential state representations across irregular frame leaps. |
| **Discrete Action Hop Space** | $\mathcal{A} \in \{1, 2, 4, 8\}$ | Hop=1 rejected by policy (0%); naturally settles into a balanced **Trimodal Strategy (2, 4, and 8 hops)**. |
| **Primary Learning Rate ($\alpha$)** | $1 \times 10^{-3}$ (Base) | Tested across sensitivity grid $\{5\times10^{-4}, 1\times10^{-3}, 2\times10^{-3}\}$ to confirm monotonic response. |
| **Temporal Discount Horizon ($\gamma$)** | $0.95$ (Base) | Evaluated across $\{0.90, 0.95, 0.99\}$; higher discount weights prioritize trajectory completeness. |
| **Exploration Schedule ($\epsilon$)** | $1.0 \to 0.1$ (Decay: $0.995$) | Decays smoothly over training episodes, enabling stable transition from exploration to exploitation. |
| **Replay Buffer Capacity** | $5,000$ transitions | Prevents sequential sample correlation bias during off-policy Q-matrix optimization. |
| **Efficiency Reward Weight ($\lambda$)** | $0.05 \sim 0.10$ | Dynamically governs desired frame acceleration without altering structural neural layers. |

---

## 6. Roadmap to Phase 2: Hierarchical Reinforcement Learning
With Phase 1 fully consolidated, benchmarked, and scientifically validated, our architecture scales into **Phase 2: Hierarchical Reinforcement Learning (H-RL) for Multi-Stream Video Surveillance**:
* **High-Level Manager Policy:** Dynamically directs computational processing across concurrent multi-camera CCTV feeds, prioritizing active intrusion or motion alert zones.
* **Low-Level Worker Policy (AdaFrame-RL):** Executes high-speed active frame hopping within targeted video streams to execute real-time human action classification at minimal hardware cost!

---
*Repository finalized and preserved under version control.*
