import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_architecture_flowchart():
    fig, ax = plt.subplots(figsize=(14, 8.5), facecolor='white', dpi=300)
    ax.set_axis_off()
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    
    # Title
    ax.text(7.0, 8.1, "AdaFrame-RL Phase 1: Joint Active Perception & Temporal Hopping Architecture", 
            ha='center', va='center', fontsize=16, fontweight='bold', color='#1e293b')
    ax.text(7.0, 7.7, "Decoupling Observation Frequency from Sequence Length via Deep Reinforcement Learning", 
            ha='center', va='center', fontsize=12, style='italic', color='#64748b')

    # Helper function for drawing boxes
    def draw_box(x, y, w, h, text_title, text_sub, facecolor, edgecolor, title_color='white', sub_color='white'):
        box = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.1,rounding_size=0.2", 
                                     facecolor=facecolor, edgecolor=edgecolor, linewidth=2.5)
        ax.add_patch(box)
        ax.text(x, y + h*0.16, text_title, ha='center', va='center', fontsize=11, fontweight='bold', color=title_color)
        ax.text(x, y - h*0.18, text_sub, ha='center', va='center', fontsize=9.5, color=sub_color, fontweight='bold')

    # Helper function for drawing arrows
    def draw_arrow(x1, y1, x2, y2, label="", rad=0.0, color='#475569'):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=2.5, color=color, connectionstyle=f"arc3,rad={rad}"))
        if label:
            mx = (x1 + x2) / 2.0; my = (y1 + y2) / 2.0
            ax.text(mx, my + 0.15, label, ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1e293b',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#f1f5f9', edgecolor='none', alpha=0.9))

    # Draw pipeline stages (Top-Down / Left-Right structured flow)
    # Stage 1: Video input
    draw_box(2.5, 6.2, 3.8, 1.1, "1. Video Feed (25 FPS)", "Sample Frame at Current Index t", "#334155", "#0f172a")
    
    # Stage 2: Spatial normalization & Backbone
    draw_box(2.5, 3.8, 3.8, 1.3, "2. Spatial Vision Backbone (F_θ)", "MobileNetV2 (112x112 Norm)\nExtracts Embedding e_t in R^1280", "#4338ca", "#312e81")
    draw_arrow(2.5, 5.65, 2.5, 4.45, "Downsample & Extract")

    # Stage 3: Temporal GRU
    draw_box(7.0, 3.8, 3.8, 1.3, "3. Temporal Memory Cell (G_ϕ)", "Gated Recurrent Unit (GRU)\nUpdates Hidden Context h_t in R^256", "#1e293b", "#3b82f6")
    draw_arrow(4.4, 3.8, 5.1, 3.8, "e_t vector")

    # GRU Recurrent Self-Loop
    ax.annotate("", xy=(7.3, 4.45), xytext=(6.7, 4.45),
                arrowprops=dict(arrowstyle="->", lw=2.0, color='#3b82f6', connectionstyle="arc3,rad=-1.2"))
    ax.text(7.0, 5.4, "Memory Retention:\nh_(t-1) State Loop", ha='center', va='center', fontsize=9, fontweight='bold', color='#2563eb',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#eff6ff', edgecolor='#bfdbfe'))

    # Stage 4: Dual Head Branching
    # Classification Head (Top Right)
    draw_box(11.5, 5.5, 4.0, 1.2, "4a. Action Recognition Head (C_ψ)", "Linear Dense Classifier in R^101\nCross-Entropy Softmax Logits", "#059669", "#047857")
    draw_arrow(8.9, 4.2, 10.2, 5.2, "h_t state")
    
    # Action Identity output badge
    ax.text(11.5, 6.8, "Prediction: Action Class\n(e.g., CricketBowling / PushUps)", ha='center', va='center', fontsize=10, fontweight='bold', color='#047857',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecfdf5', edgecolor='#10b981', lw=2))
    draw_arrow(11.5, 6.1, 11.5, 6.4)

    # Policy Head (Bottom Right)
    draw_box(11.5, 2.3, 4.0, 1.2, "4b. DQN Policy Head (Q_ω)", "Multi-Layer Perceptron in R^4\nEstimates Q-Values over Hop Stride", "#b45309", "#9a3412")
    draw_arrow(8.9, 3.4, 10.2, 2.6, "h_t state")

    # Hop Action Output badge
    ax.text(11.5, 0.8, "Action: Temporal Stride Leap\nA_t in {1, 2, 4, 8} Frames", ha='center', va='center', fontsize=10, fontweight='bold', color='#9a3412',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffbeb', edgecolor='#f59e0b', lw=2))
    draw_arrow(11.5, 1.7, 11.5, 1.15)

    # Active feedback loop back to Video feed
    ax.annotate("", xy=(2.5, 5.65), xytext=(9.5, 0.8),
                arrowprops=dict(arrowstyle="->", lw=3.0, color='#e11d48', linestyle="--", connectionstyle="bar,angle=180,fraction=-0.15"))
    ax.text(5.5, 1.1, "ACTIVE PERCEPTION SKIP LOOP: Vault directly to Frame Index (t + A_t)\nBypassing & Eliminating 100% Compute for all intermediate redundant frames!", 
            ha='center', va='center', fontsize=10.5, fontweight='bold', color='#e11d48',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff1f2', edgecolor='#f43f5e', lw=2))

    plt.tight_layout()
    out_dir = os.path.join("Phase_1", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "architecture_pipeline.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f" -> [Figure Saved Successfully] {out_path}")

if __name__ == "__main__":
    generate_architecture_flowchart()
