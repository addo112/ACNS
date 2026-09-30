"""Test script for the ACNS Engine"""
import sys
sys.path.insert(0, '.')
from acns_engine import ACNSEngine

print("=" * 50)
print("  ACNS Engine Test Suite")
print("=" * 50)

# Test 1: Engine creation
print("\n[Test 1] Creating IEEE-14 engine...")
engine = ACNSEngine(network_type='ieee14')
print(f"  Buses: {engine.n}")
print(f"  Gen buses: {engine.topo['gen_buses']}")
print(f"  Lines: {len(engine.topo['lines'])}")

# Test 2: Step simulation
print("\n[Test 2] Running 20 steps...")
for i in range(20):
    state = engine.step()

m = state['metrics']
print(f"  Time: {state['time']:.4f}s")
print(f"  Step: {state['step']}")
print(f"  Global Error J: {m['global_error']:.6f}")
print(f"  Freq Dev Max: {m['freq_dev_max']:.6f} Hz")
print(f"  Voltage RMSE: {m['voltage_rmse']:.6f} p.u.")
print(f"  Pheromone Mean: {m['pheromone_mean']:.6f}")
print(f"  Convergence Ratio: {m['convergence_ratio']:.6f}")

# Test 3: Disturbance injection
print("\n[Test 3] Injecting load disturbance...")
result = engine.inject_disturbance('load', 0.15)
print(f"  Affected buses: {result['affected_buses']}")
J_before = engine.metrics_history['global_error'][-1]

for i in range(50):
    engine.step()

J_after = engine.metrics_history['global_error'][-1]
print(f"  J before recovery: {J_before:.6f}")
print(f"  J after 50 steps: {J_after:.6f}")
print(f"  Recovery: {((J_before - J_after) / max(J_before, 1e-10)) * 100:.1f}%")

# Test 4: Pheromone map
print("\n[Test 4] Pheromone map...")
pm = engine.get_pheromone_map()
print(f"  Edges: {len(pm['edges'])}")
print(f"  Matrix size: {len(pm['matrix'])}x{len(pm['matrix'][0])}")
print(f"  Full mean: {pm['full_mean']:.6f}")
print(f"  Full max: {pm['full_max']:.6f}")

# Test 5: Metrics history
print("\n[Test 5] Metrics history...")
mh = engine.get_metrics_history()
print(f"  Time points: {len(mh['time'])}")
print(f"  Error range: [{min(mh['global_error']):.6f}, {max(mh['global_error']):.6f}]")

# Test 6: Network state
print("\n[Test 6] Network state snapshot...")
ns = engine.get_network_state()
print(f"  Buses: {ns['n_buses']}")
print(f"  Agents: {ns['n_agents']}")
V_values = [b['V'] for b in ns['buses']]
w_values = [b['w'] for b in ns['buses']]
print(f"  V range: [{min(V_values):.4f}, {max(V_values):.4f}]")
print(f"  w range: [{min(w_values):.4f}, {max(w_values):.4f}]")

# Test 7: Comparison
print("\n[Test 7] Running ACNS vs Droop vs Consensus comparison...")
comp = engine.run_comparison(steps=200)
print(f"  ACNS final J: {comp['summary']['acns']['J_final']:.6f}")
print(f"  Droop final J: {comp['summary']['droop']['J_final']:.6f}")
print(f"  Consensus final J: {comp['summary']['consensus']['J_final']:.6f}")
print(f"  ACNS improvement: {comp['summary']['acns']['improvement']:.1f}%")
print(f"  ACNS t5%%: {comp['summary']['acns']['t5_percent']}")

print("\n" + "=" * 50)
print("  ALL TESTS PASSED!")
print("=" * 50)
