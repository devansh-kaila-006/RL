"""
AdaFrame-RL Phase 1: Interactive Demonstration & Simulation Entrypoint
Run this script directly from the project root:
  python demo_phase1_simulation.py --class-name Basketball
  python demo_phase1_simulation.py --class-name BenchPress
  python demo_phase1_simulation.py --class-name PlayingGuitar
  python demo_phase1_simulation.py --interactive
"""
import os
import sys

# Forward directly to Phase_1/scripts/simulate_phase1_demo.py
if __name__ == "__main__":
    script_path = os.path.join(os.path.dirname(__file__), "Phase_1", "scripts", "simulate_phase1_demo.py")
    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, script_path, 'exec'))
