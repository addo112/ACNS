"""Quick API integration test for ACNS server"""
import urllib.request
import json
import time

BASE = 'http://localhost:5000'

def api_get(path):
    resp = urllib.request.urlopen(f'{BASE}{path}')
    return json.loads(resp.read())

def api_post(path, data=None):
    req = urllib.request.Request(
        f'{BASE}{path}',
        data=json.dumps(data or {}).encode(),
        headers={'Content-Type': 'application/json'}
    )
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())

print("ACNS API Integration Test")
print("=" * 40)

# 1. Networks
r = api_get('/api/networks')
assert r['status'] == 'success'
print(f"[PASS] Networks: {len(r['networks'])} available")

# 2. Create
r = api_post('/api/simulation/create', {'network': 'ieee14'})
assert r['status'] == 'success'
print(f"[PASS] Create: {r['state']['n_buses']} buses")

# 3. Step
r = api_post('/api/simulation/step')
assert r['status'] == 'success'
print(f"[PASS] Step: t={r['state']['time']}, J={r['state']['metrics']['global_error']:.6f}")

# 4. Multiple steps
for _ in range(20):
    r = api_post('/api/simulation/step')
print(f"[PASS] 20 steps: t={r['state']['time']:.4f}, J={r['state']['metrics']['global_error']:.6f}")

# 5. Disturbance
r = api_post('/api/simulation/disturbance', {'type': 'load', 'magnitude': 0.15})
assert r['status'] == 'success'
print(f"[PASS] Disturbance: {r['message']}")

# 6. State
r = api_get('/api/simulation/state')
assert r['status'] == 'success'
buses = r['state']['buses']
V_min = min(b['V'] for b in buses)
V_max = max(b['V'] for b in buses)
print(f"[PASS] State: V=[{V_min:.4f}, {V_max:.4f}]")

# 7. Metrics
r = api_get('/api/simulation/metrics')
assert r['status'] == 'success'
print(f"[PASS] Metrics: {len(r['metrics']['time'])} data points")

# 8. Pheromone
r = api_get('/api/simulation/pheromone')
assert r['status'] == 'success'
print(f"[PASS] Pheromone: {len(r['pheromone']['edges'])} edges, mean={r['pheromone']['full_mean']:.4f}")

# 9. Reset
r = api_post('/api/simulation/reset')
assert r['status'] == 'success'
print(f"[PASS] Reset: {r['message']}")

# 10. Export
api_post('/api/simulation/create', {'network': 'ieee14'})
for _ in range(5):
    api_post('/api/simulation/step')
r = api_get('/api/simulation/export')
assert 'metrics' in r and 'final_state' in r
print(f"[PASS] Export: {len(json.dumps(r))} bytes")

# 11. Dashboard
resp = urllib.request.urlopen(f'{BASE}/')
html = resp.read().decode()
assert 'ACNS Power Network Controller' in html
assert 'Chart.js' in html or 'chart.js' in html
assert 'd3.v7' in html or 'D3.js' in html
print(f"[PASS] Dashboard HTML: {len(html)} bytes served")

print()
print("=" * 40)
print("ALL 11 API TESTS PASSED!")
print("=" * 40)
