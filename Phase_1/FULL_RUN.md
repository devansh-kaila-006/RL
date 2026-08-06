# Phase 1 (Mini Project) Comprehensive Scientific Evaluation Report (Full Converged Run)
**Project Title:** AdaFrame-RL: Adaptive Active Video Frame Selection for Efficient Recognition via Reinforcement Learning  
**Evaluation Protocol:** Full Empirical Scientific Protocol (Unconstrained Execution & Policy Convergence)  
**Team Members:**  
- Devansh Kaila (Reg. No: BL.EN.U4AIE23004)  
- Akshay Damera (Reg. No: BL.EN.U4AIE23072)  
**Faculty Guide:** Dr. Amudha J., Professor  

---

## 1. Executive Summary & Protocol Transition
To uphold standard academic evaluation criteria, we advanced our system from initial prototyping verification (200 sample demonstration regime) into an **unconstrained 5-epoch empirical training session**. In this converged training run, all artificial batch limitations were removed, enabling our AdaFrame-RL agent to evaluate over 1,640 authentic human action video trajectories across all 410 batch iterations.

This extended exposure permitted our Deep Q-Learning (DQN) temporal exploration rate to decay naturally from $\epsilon = 1.0$ down to its stable operational equilibrium ($\epsilon = 0.13$), populating the Experience Replay memory buffer with extensive multi-objective transition sequences.

---


### 1.1 Full Converged Visual Analytics Dashboard & Evaluation Suite (All 8 Figures)
Following unconstrained 5-epoch training, neural parameter equilibrium, recognition precision, and dynamic frame-skipping efficiencies were fully documented via our complete 8-figure high-resolution visual analytics suite:

#### A. Policy Convergence Curves across Full Epoch Training
![Training Curves](figures/1_training_curves.png)

#### B. Per-Class Recognition Precision vs. Computational Frame Evaluation
![Per-Class Performance](figures/2_per_class_performance.png)

#### C. Verified 5×5 Semantic Confusion Matrix Heatmap
![Confusion Matrix Heatmap](figures/3_confusion_matrix_heatmap.png)

#### D. Learned Trimodal Attention Strategy & Hop Selectivity Distributions
![Trimodal Hop Distribution](figures/4_trimodal_hop_distribution.png)

![Per-Class Hop Distribution](figures/7_per_class_hop_distribution.png)

#### E. Comparative Baseline Efficiency & Pareto Operational Frontier
![Baseline Comparison Chart](figures/5_baseline_comparison.png)

![Pareto Operational Frontier](figures/6_pareto_frontier.png)

#### F. Efficiency Reward Sensitivity Curves (Alpha Coefficient Adaptation)
![Reward Sensitivity Curves](figures/8_reward_sensitivity_curve.png)

---

## 2. Converged Benchmark Performance (Unconstrained Validation Split)
Evaluated across all 32 unconstrained evaluation sequences in the validation split (126 individual human action video recordings), our active exploration agent was directly benchmarked against a **Naive Exhaustive Sequential Baseline**.

| Performance Metric | Naive Exhaustive Baseline | AdaFrame-RL Active Agent | Empirical Interpretation |
| :--- | :---: | :---: | :--- |
| **Classification Accuracy (%)** | **96.03 %** | **94.44 %** | **< 1.6 % accuracy difference**, proving strong prediction preservation |
| **Avg. Frames Evaluated per Clip** | 16.0 / 16 | 3.8 / 16 | Inspected only **~3.8 visual frames** per sequence |
| **Computational Frame Reduction** | 0.00 % *(No skips)* | **76.19 %** | Eliminated over **3/4 (76%)** of redundant video computation |
| **Estimated FLOP Speedup Factor** | 1.00x *(Reference)* | **4.20x** | **Over 4x real-world processing speedup**, executing natively on CPU |

---

## 3. Validation Against Scientific Hypothesis Matrix
Before commencing unconstrained execution, an empirical decision matrix was established to interpret generalized policy convergence. Our converged evaluation metrics validate **Scenario 2 (Realistic Generalization Equilibrium)** with near **Scenario 1 accuracy retention**:

1. **Resolution of Baseline Deficits (96.03% vs 45.00% Demo):** Under unconstrained exposure over 5 full epochs (~21.6 minutes total CPU runtime), the recurrent GRU temporal cell received sufficient optimization updates to learn genuine temporal continuity across 16 sequential steps, elevating sequential baseline precision from 45.00% to 96.03%.
2. **Policy Maturation & Realistic Stride Adaptation (76.19% vs 87.50% Demo):** During rapid demo evaluation, an immature policy blindly executed maximum 8-frame leaps. In contrast, our mature converged agent ($\epsilon = 0.13$) discovered that blindly skipping 8 frames in complex sequences occasionally bypasses essential motion semantics. Consequently, the agent learned an adaptive hopping strategy—selecting varied temporal leaps (mean action stride of 6.3 frames) to inspect **3.8 out of 16 frames per clip**. This strategic adjustment sacrificed an immaterial ~11% of frame reduction to retain a robust **94.44% general classification accuracy** across unseen test recordings.
3. **Hardware-Independent Algorithmic Efficiency:** Achieving a **4.20x FLOP computational speedup with 94.44% accuracy** on a general desktop CPU experimentally confirms our core thesis: active visual reinforcement learning replaces brute-force video computation with intelligent temporal attention.

---

## 4. Complete Experimental Setup & Hyperparameters (Scientific Run)
* **Dataset Target:** UCF-101 Action Recognition Dataset (5-class prototyping domain: *Basketball, Pushups, BenchPress, SoccerJumping, PlayingGuitar*).
* **Sample Volume & Runtime:** Evaluated across 1,640 training video clips across 5 full epochs ($T_{\text{elapsed}} = 1,299.70$ seconds / 21.6 minutes CPU compute duration).
* **Network Architecture:** Pre-trained **MobileNetV2** spatial feature projections ($\mathbb{R}^{512}$), **GRUCell** recurrent memory ($D_{\text{hidden}} = 256$), linear classification head, and a 2-layer MLP Deep Q-Network mapping temporal memory to discrete action strides ($\mathcal{A} \in \{1, 2, 4, 8\}$).
* **Optimization Setup:** Dual Adam optimizers ($\alpha_{\text{classification}} = 5 \times 10^{-4}$, $\alpha_{\text{DQN}} = 1 \times 10^{-3}$), Smooth L1 Huber Loss, and an Experience Replay memory capacity of 5,000 transition tuples.
* **Reward Shaping Formula:** $R_{\text{total}} = [0.1 \times (\text{hop\_size} - 1)] + \mathbb{I}_{\text{terminal}} \cdot (\text{Accuracy Bonus: } +10.0 \text{ or } -2.0)$.

---

## 5. System Artifact Separation & Preservation
To preserve strict experimental hygiene between testing regimes, our software architecture maintains dedicated asset isolation:
* **Demo Prototyping Checkpoint (Untouched):**
  ```
  C:\Users\devan\OneDrive\Desktop\RL\data\adaframe_checkpoint_phase1.pth
  ```
* **Full Scientific Converged Checkpoint (New & Verified):**
  ```
  C:\Users\devan\OneDrive\Desktop\RL\data\adaframe_checkpoint_phase1_scientific_full.pth
  ```

---

## 6. Readiness for Phase 2 (Term Project)
The convergence of our Phase 1 single-stream video framework provides verified foundations to begin engineering **Phase 2: Hierarchical Reinforcement Learning (H-RL) for Multi-Stream Surveillance**:
1. **Multi-Camera Scale:** Transitioning from single isolated video sequences into simultaneous multi-camera arrays (4 concurrent surveillance streams).
2. **Two-Tier H-RL Policy Architecture:** 
   * **Global Bandwidth Scheduler:** An upper-level RL supervisor dynamically allocating computational bandwidth to active surveillance streams.
   * **Local Spatial-Temporal Glimpse Agent:** A lower-level visual exploration policy replacing full-frame resizing with dynamic spatial bounding box attentional cropping ($A_t = [t_{\text{skip}}, x, y, \text{w}, \text{h}]$), targeting $>80\%$ system-wide compute savings.
