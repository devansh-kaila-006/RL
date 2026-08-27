"""Pre-flight gradient-flow probe: verifies Manager REINFORCE and Worker DQN gradients actually propagate."""
import os
import sys
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.hierarchical.stream_synthesizer import MultiStreamSynthesizer
from src.hierarchical.manager_policy import ManagerPolicy
from src.hierarchical.worker_agent import WorkerAgent, ReplayBuffer

torch.manual_seed(0)
K = 4
device = torch.device("cpu")

# --- Synthetic infinite stream source (fast, no video decoding) ---
dummy = [(torch.randn(40, 3, 112, 112), i % 101, "x") for i in range(8)]
def infinite():
    while True:
        for item in dummy:
            yield item

synth = MultiStreamSynthesizer(infinite(), K_streams=K, max_sequence_length=40)
manager = ManagerPolicy(K_streams=K)
worker = WorkerAgent(num_classes=101)

# ================= 1. Manager REINFORCE gradient =================
frames, labels, entropies = synth.get_current_frames()
budgets, log_probs = manager.allocate_discrete_budgets(entropies.unsqueeze(0))

# replicate one b_step of worker processing
h_worker = worker.init_hidden(batch_size=K, device=device)
prev_glimpses = torch.zeros(K, dtype=torch.long)
h_new, logits, q_values = worker(frames, h_worker, prev_glimpses)
labels_t = torch.tensor(labels)
loss_cls = F.cross_entropy(logits, labels_t)

final_preds = torch.argmax(logits, dim=-1)
manager_reward = (final_preds == labels_t).float() - 0.05 * (budgets.squeeze(0).float() / 20.0)
loss_manager = -(log_probs.squeeze(0) * manager_reward.detach()).mean()
loss_manager.backward()

m_grad = manager.budget_head.weight.grad
m_ok = m_grad is not None and m_grad.abs().sum().item() > 0
print(f"[{'PASS' if m_ok else 'FAIL'}] Manager REINFORCE gradient flows: |grad| sum = "
      f"{m_grad.abs().sum().item():.6f}" if m_grad is not None else "[FAIL] Manager grad is None")

# ================= 2. Worker DQN gradient =================
worker.zero_grad()
buffer = ReplayBuffer(capacity=1000)
# Prefill with synthetic transitions
for _ in range(300):
    s = torch.randn(256); ns = torch.randn(256)
    buffer.push(s, torch.randint(0, 24, (1,)).item(), float(torch.rand(1) > 0.5), ns, False)

state, action, reward, next_state, done = buffer.sample(64)
q_s = worker.policy_net(state)
q_s_a = q_s.gather(1, action.unsqueeze(1)).squeeze(1)
with torch.no_grad():
    target_q = reward + 0.99 * worker.policy_net(next_state).max(1)[0] * (1 - done)
loss_q = F.mse_loss(q_s_a, target_q)
loss_q.backward()

p_grad = worker.policy_net[2].weight.grad
p_ok = p_grad is not None and p_grad.abs().sum().item() > 0
print(f"[{'PASS' if p_ok else 'FAIL'}] Worker DQN (policy_net) gradient flows: |grad| sum = {p_grad.abs().sum().item():.6f}")

# ================= 3. ReplayBuffer round-trip =================
s2, a2, r2, ns2, d2 = buffer.sample(32)
shapes_ok = s2.shape == (32, 256) and a2.shape == (32,) and r2.dtype == torch.float32
print(f"[{'PASS' if shapes_ok else 'FAIL'}] ReplayBuffer sample shapes: state {tuple(s2.shape)}, action {tuple(a2.shape)}")

# ================= 4. Manager budget semantics =================
b, lp = manager.allocate_discrete_budgets(entropies.unsqueeze(0))
in_range = (b.min() >= 1).item() and (b.max() <= 20).item()
print(f"[{'PASS' if in_range else 'FAIL'}] Manager budgets in [1, 20]: {b.flatten().tolist()}")

# ================= 5. grid_sample glimpse geometry =================
frame = torch.zeros(1, 3, 112, 112)
frame[:, :, :56, :56] = 1.0  # TL quadrant = 1
crop = worker.glimpse_extractor(frame, torch.tensor([1]))  # TL glimpse
tl_mean = crop.mean().item()
# Center crop overlaps TL quadrant in rows/cols 28-55 => expected mean exactly 0.25
crop_c = worker.glimpse_extractor(frame, torch.tensor([5]))
c_mean = crop_c.mean().item()
# BR quadrant should be pure 0 for both crops
frame2 = torch.zeros(1, 3, 112, 112)
frame2[:, :, 56:, 56:] = 1.0  # BR quadrant = 1
crop_tl2 = worker.glimpse_extractor(frame2, torch.tensor([1])).mean().item()
crop_c2 = worker.glimpse_extractor(frame2, torch.tensor([5])).mean().item()
geo_ok = (abs(tl_mean - 1.0) < 1e-3 and abs(c_mean - 0.25) < 1e-3
          and crop_tl2 < 1e-3 and abs(crop_c2 - 0.25) < 1e-3)
print(f"[{'PASS' if geo_ok else 'FAIL'}] Glimpse geometry: TL->TL={tl_mean:.4f} (1.0), TL->Center={c_mean:.4f} (0.25 expected overlap), BR->TL={crop_tl2:.4f} (0.0), BR->Center={crop_c2:.4f} (0.25)")

print()
if m_ok and p_ok and shapes_ok and in_range and geo_ok:
    print("PREFLIGHT: ALL GRADIENT/LOGIC CHECKS PASSED - safe to run")
else:
    print("PREFLIGHT: FAILURES DETECTED - do not start the long run")
