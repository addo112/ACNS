# ⚡ ACNS Power Network Controller

**Ant Colony Neural Swarm Coordination for Self-Organizing Microgrids**

A real-time web application implementing the ACNS framework for decentralized and self-organizing coordination of inverter-based power networks.

> Based on the IEEE paper: *"Ant Colony–Driven Neural Swarm Coordination for Self-Organizing Microgrids"* — Addo, Kabeya, Ojo, Loji (Durban University of Technology, 2026)

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## 📐 What It Implements

### Core ACNS Algorithm (Algorithm 1)
- **Pheromone Dynamics** (Eq. 6–9): Evaporation, reinforcement, and normalized coordination weights
- **Neural Control Policies** (Eq. 21–25): Two-hidden-layer feedforward networks (16 neurons/layer) with online SGD
- **Swing Equation** (Eq. 5): Frequency dynamics with inertia and damping
- **Error Dynamics** (Eq. 30): Lyapunov-stable coupling via symmetric pheromone weights
- **Global Regulation Cost** (Eq. 11): Combined frequency and voltage error metric

### Supported Networks
| Network | Buses | Description |
|---------|-------|-------------|
| IEEE 14-Bus | 14 | Small test network |
| IEEE 30-Bus | 30 | Medium test network |
| IEEE 118-Bus | 118 | Large-scale benchmark (used in paper) |

### Comparison Baselines
- **Conventional Droop Control** — Local P-f and Q-V droop regulation
- **Consensus-Based Control** — Distributed neighbor-averaging coordination
- **ACNS** — Proposed pheromone + neural swarm framework

---

## 🖥️ Dashboard Features

- 🌐 **Interactive Network Topology** — D3.js force-directed graph with voltage-colored nodes
- 📈 **Real-Time Frequency Response** — Live tracking of ω deviation from 60 Hz
- 📊 **Voltage Profile** — Color-coded bus voltages with operational bounds
- 🔥 **Pheromone Heatmap** — Canvas-rendered τ_ij matrix visualization
- 📉 **Convergence Plot** — Log-scale global error J(t) tracking
- 🎛️ **Parameter Tuning** — Adjust ρ, κ, η_θ, α, β, γ, δ in real-time
- ⚡ **Disturbance Injection** — Load, PV variation, and bus fault scenarios
- 🔄 **Comparison Mode** — Overlay ACNS vs Droop vs Consensus traces

---

## 🔌 REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/networks` | GET | List available IEEE networks |
| `/api/simulation/create` | POST | Create new simulation |
| `/api/simulation/step` | POST | Advance one step |
| `/api/simulation/run` | POST | Run N steps (WebSocket streaming) |
| `/api/simulation/stop` | POST | Stop running simulation |
| `/api/simulation/disturbance` | POST | Inject disturbance |
| `/api/simulation/state` | GET | Get current network state |
| `/api/simulation/metrics` | GET | Get full metrics history |
| `/api/simulation/pheromone` | GET | Get pheromone map |
| `/api/simulation/compare` | POST | Run baseline comparison |
| `/api/simulation/reset` | POST | Reset simulation |
| `/api/simulation/export` | GET | Export results as JSON |

---

## ⚙️ Default Parameters (Paper Section VII)

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Reference frequency | ω_ref | 60 Hz | Nominal system frequency |
| Reference voltage | V_ref | 1.0 p.u. | Nominal bus voltage |
| Evaporation rate | ρ | 0.05 | Pheromone decay rate |
| Reinforcement gain | κ | 0.1 | Pheromone deposit scaling |
| Learning rate | η_θ | 0.01 | Neural policy learning rate |
| Voltage penalty | γ | 2.0 | Voltage mismatch weight |
| Frequency penalty | δ | 2.0 | Frequency mismatch weight |
| Pheromone influence | α | 1.0 | Exploration-exploitation |
| Heuristic influence | β | 1.0 | Exploration-exploitation |

---

## 📊 Results (Matching Paper Table I)

| Method | J(final) | Improvement |
|--------|----------|-------------|
| Droop | 0.044961 | — |
| Consensus | 0.005177 | 88.5% |
| **ACNS** | **0.000011** | **99.98%** |

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask, Flask-SocketIO, PyTorch, pandapower, NumPy
- **Frontend**: Vanilla JavaScript (ES6+), Chart.js, D3.js, HTML5 Canvas
- **Design**: Dark theme, glassmorphism, responsive CSS Grid

---

## 📄 License

This project implements the research from IEEE paper 979-8-3195-1782-1/26.
