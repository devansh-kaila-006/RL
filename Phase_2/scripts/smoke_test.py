"""Quick structural smoke test for Phase 2 H-RL components (2 macro-steps only)."""
import os
import sys
import traceback

# Add repo root to path (scripts run directly get their own dir on sys.path, not the repo root)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

def check(name, fn):
    print(f"\n[CHECK] {name}...")
    try:
        fn()
        print(f"[PASS] {name}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

results = {}

# 1. Imports
def test_imports():
    from src.hierarchical.manager_policy import ManagerPolicy
    from src.hierarchical.worker_agent import WorkerAgent
    from src.hierarchical.stream_synthesizer import MultiStreamSynthesizer
results["imports"] = check("Module imports", test_imports)

# 2. Instantiate models
def test_instantiate():
    from src.hierarchical.manager_policy import ManagerPolicy
    from src.hierarchical.worker_agent import WorkerAgent
    m = ManagerPolicy(K_streams=4)
    w = WorkerAgent(num_classes=101)
    # Check the attribute scripts rely on
    assert hasattr(w, "forward_step") or hasattr(w, "forward"), "No forward method"
    print(f"  WorkerAgent methods: {[x for x in dir(w) if 'forward' in x or 'decode' in x]}")
results["instantiate"] = check("Model instantiation", test_instantiate)

# 3. Manager forward pass
def test_manager():
    import torch
    from src.hierarchical.manager_policy import ManagerPolicy
    m = ManagerPolicy(K_streams=4)
    entropies = torch.rand(1, 4)
    budgets = m.allocate_discrete_budgets(entropies, max_total_budget=20)
    assert budgets.shape == (1, 4), f"Bad budget shape {budgets.shape}"
    print(f"  Budgets: {budgets.tolist()}")
results["manager"] = check("ManagerPolicy forward", test_manager)

# 4. Worker forward pass (correct method name: forward)
def test_worker_forward():
    import torch
    from src.hierarchical.worker_agent import WorkerAgent
    w = WorkerAgent(num_classes=101)
    frames = torch.rand(4, 3, 224, 224)
    h = w.init_hidden(batch_size=4, device=torch.device("cpu"))
    glimpses = [0, 0, 0, 0]
    h_new, logits, q_values = w(frames, h, glimpses)
    assert q_values.shape == (4, 24), f"Bad q shape {q_values.shape}"
    hop, g = w.decode_action(10)
    print(f"  Q-values: {q_values.shape}, decode_action(10) -> hop={hop}, glimpse={g}")
results["worker_forward"] = check("WorkerAgent.forward (vectorized)", test_worker_forward)

# 5. The call the 3 scripts actually make: forward_step
def test_worker_forward_step():
    import torch
    from src.hierarchical.worker_agent import WorkerAgent
    w = WorkerAgent(num_classes=101)
    frames = torch.rand(1, 3, 224, 224)
    h = w.init_hidden(batch_size=1, device=torch.device("cpu"))
    h_new, logits, q_values = w.forward_step(frames, h, 0)
results["worker_forward_step"] = check("WorkerAgent.forward_step (as scripts call it)", test_worker_forward_step)

# 6. Synthesizer with a LIST (as run_1_epoch_test / profiler / train_hrl pass it)
def test_synth_list():
    import torch
    from src.hierarchical.stream_synthesizer import MultiStreamSynthesizer
    dummy = [(torch.randn(100, 3, 112, 112), i % 101, "x") for i in range(10)]
    s = MultiStreamSynthesizer(dummy, K_streams=4, max_sequence_length=150)
results["synth_list"] = check("MultiStreamSynthesizer fed a raw list", test_synth_list)

# 7. Synthesizer with a proper infinite iterator
def test_synth_iterator():
    import torch
    from src.hierarchical.stream_synthesizer import MultiStreamSynthesizer
    dummy = [(torch.randn(100, 3, 112, 112), i % 101, "x") for i in range(10)]
    def infinite():
        while True:
            for item in dummy:
                yield item
    s = MultiStreamSynthesizer(iter(infinite()), K_streams=4, max_sequence_length=150)
    frames, labels, entropies = s.get_current_frames()
    print(f"  Frames: {frames.shape}, entropies: {entropies.shape}")
    s.step_streams([4, 4, 4, 4])
results["synth_iterator"] = check("MultiStreamSynthesizer fed iter(infinite())", test_synth_iterator)

# 8. End-to-end: 2 macro-steps of the H-RL loop (using only APIs that exist)
def test_e2e():
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    from src.hierarchical.manager_policy import ManagerPolicy
    from src.hierarchical.worker_agent import WorkerAgent
    from src.hierarchical.stream_synthesizer import MultiStreamSynthesizer

    K = 2
    device = torch.device("cpu")
    dummy = [(torch.randn(20, 3, 112, 112), i % 101, "x") for i in range(4)]
    def infinite():
        while True:
            for item in dummy:
                yield item
    synth = MultiStreamSynthesizer(iter(infinite()), K_streams=K, max_sequence_length=20)
    manager = ManagerPolicy(K_streams=K)
    worker = WorkerAgent(num_classes=101)
    m_opt = optim.Adam(manager.parameters(), lr=1e-4)
    w_opt = optim.Adam(worker.parameters(), lr=1e-4)
    h = worker.init_hidden(batch_size=K, device=device)
    prev_glimpses = [0] * K

    for step in range(2):
        m_opt.zero_grad(); w_opt.zero_grad()
        frames, labels, entropies = synth.get_current_frames()
        budgets = manager.allocate_discrete_budgets(entropies.unsqueeze(0), max_total_budget=20)
        glimpse_tensor = torch.tensor(prev_glimpses, dtype=torch.long)
        h_new, logits, q_values = worker(frames, h, glimpse_tensor)
        loss = F.cross_entropy(logits, torch.tensor(labels))
        loss.backward()
        w_opt.step(); m_opt.step()
        h = h_new.detach()
        action_indices = torch.argmax(q_values, dim=-1)
        hops = []
        for k in range(K):
            hop, g = worker.decode_action(action_indices[k].item())
            hops.append(hop); prev_glimpses[k] = g
        synth.step_streams(hops)
        print(f"  step {step+1}: loss={loss.item():.4f}, hops={hops}, budgets={budgets.flatten().tolist()}")
results["e2e"] = check("End-to-end 2-step H-RL loop", test_e2e)

print("\n" + "=" * 60)
passed = sum(results.values())
print(f"RESULTS: {passed}/{len(results)} checks passed")
for k, v in results.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
