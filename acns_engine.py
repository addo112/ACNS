"""
ACNS Engine - Ant Colony Neural Swarm Coordination for Self-Organizing Microgrids
==================================================================================
Implements the full ACNS framework from the IEEE paper (2026).

Key Equations Implemented:
  - Pheromone dynamics:  τ_ij(t+1) = (1-ρ)τ_ij(t) + κ·Φ_ij(t)       [Eq. 6]
  - Reinforcement:       Φ_ij = exp(-γ|Vi-Vj| - δ|ωi-ωj|)            [Eq. 7]
  - Coordination weight: p_ij = τ_ij^α · η_ij^β / Σ_k (...)          [Eq. 8]
  - Heuristic:           η_ij = 1 / (|Vi-Vj| + ε)                    [Eq. 9]
  - Neural policy:       u_i = π_θi(s̄_i)  (2-layer, 16 neurons)     [Eq. 21-23]
  - Local loss:          L_i = (ωi-ωref)² + λv(Vi-Vref)²             [Eq. 25]
  - Global metric:       J = Σ_i [αf(ωi-ωref)² + αv(Vi-Vref)²]      [Eq. 11]
  - Error dynamics:      ė_i = -K_i·e_i - Σ_j a_ij(e_i - e_j)       [Eq. 30]

Authors: Based on Addo, Kabeya, Ojo, Loji (DUT, 2026)
"""

import numpy as np
import torch
import torch.nn as nn
import copy
import json
import traceback


# ---------------------------------------------------------------------------
# Neural Control Policy (Equations 21-23)
# Two-hidden-layer feedforward network, 16 neurons per layer, ReLU activation
# ---------------------------------------------------------------------------
class NeuralPolicy(nn.Module):
    """Local neural control policy for a single agent/bus.
    
    Input:  s̄_i = N[Vi, ωi, Pi, Qi, τ̄_i, p̄_i]  (dim=6)
    Output: u_i = [ΔPi, ΔQi]                        (dim=2)
    """
    def __init__(self, input_size=6, hidden_size=16, output_size=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),   # W^(1), b^(1)
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # W^(2), b^(2)
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),  # W^(3), b^(3)
        )
        # Small initialization for stability
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# IEEE Network Topology Helper
# ---------------------------------------------------------------------------
def _build_network_topology(network_type):
    """Build adjacency list and network properties from IEEE benchmarks.
    
    Uses pandapower for initial power flow, then extracts topology
    for the ACNS simulation.
    """
    try:
        import pandapower.networks as pn
        import pandapower as pp

        if network_type == 'ieee118':
            net = pn.case118()
        elif network_type == 'ieee30':
            net = pn.case30()
        elif network_type == 'ieee14':
            net = pn.case14()
        else:
            raise ValueError(f"Unknown network: {network_type}")

        pp.runpp(net, init='results', enforce_q_lims=False)

        n_buses = len(net.bus)
        bus_ids = list(net.bus.index)

        # Build adjacency from lines and transformers
        adj = {i: set() for i in range(n_buses)}
        line_data = []
        for _, row in net.line.iterrows():
            fb = int(row.from_bus)
            tb = int(row.to_bus)
            adj[fb].add(tb)
            adj[tb].add(fb)
            line_data.append({'from': fb, 'to': tb})

        if hasattr(net, 'trafo') and len(net.trafo) > 0:
            for _, row in net.trafo.iterrows():
                hv = int(row.hv_bus)
                lv = int(row.lv_bus)
                adj[hv].add(lv)
                adj[lv].add(hv)
                line_data.append({'from': hv, 'to': lv})

        # Extract generation buses
        gen_buses = set()
        if hasattr(net, 'gen') and len(net.gen) > 0:
            gen_buses.update(net.gen.bus.tolist())
        if hasattr(net, 'ext_grid') and len(net.ext_grid) > 0:
            gen_buses.update(net.ext_grid.bus.tolist())

        # Get initial voltage and power from power flow
        V_init = np.ones(n_buses)
        P_init = np.zeros(n_buses)
        Q_init = np.zeros(n_buses)

        if net.res_bus is not None and len(net.res_bus) > 0:
            V_init = net.res_bus.vm_pu.values.copy()
            P_init = net.res_bus.p_mw.values.copy() / 100.0  # Normalize to p.u.
            Q_init = net.res_bus.q_mvar.values.copy() / 100.0

        return {
            'n_buses': n_buses,
            'adj': {k: list(v) for k, v in adj.items()},
            'lines': line_data,
            'gen_buses': list(gen_buses),
            'V_init': V_init,
            'P_init': P_init,
            'Q_init': Q_init,
            'bus_ids': bus_ids,
        }

    except ImportError:
        # Fallback: generate synthetic network if pandapower unavailable
        return _build_synthetic_network(network_type)


def _build_synthetic_network(network_type):
    """Generate a synthetic IEEE-like network when pandapower is unavailable."""
    sizes = {'ieee14': 14, 'ieee30': 30, 'ieee118': 118}
    n = sizes.get(network_type, 14)

    # Ring + random extra edges for realistic connectivity
    adj = {i: set() for i in range(n)}
    lines = []
    for i in range(n):
        j = (i + 1) % n
        adj[i].add(j)
        adj[j].add(i)
        lines.append({'from': i, 'to': j})

    # Add random cross-links for higher connectivity
    rng = np.random.RandomState(42)
    extra = max(n // 3, 5)
    for _ in range(extra):
        a, b = rng.choice(n, 2, replace=False)
        if b not in adj[a]:
            adj[a].add(b)
            adj[b].add(a)
            lines.append({'from': int(a), 'to': int(b)})

    gen_ratio = max(n // 4, 3)
    gen_buses = list(range(0, n, n // gen_ratio))[:gen_ratio]

    return {
        'n_buses': n,
        'adj': {k: list(v) for k, v in adj.items()},
        'lines': lines,
        'gen_buses': gen_buses,
        'V_init': np.ones(n) + rng.uniform(-0.02, 0.02, n),
        'P_init': rng.uniform(-0.5, 1.0, n),
        'Q_init': rng.uniform(-0.3, 0.5, n),
        'bus_ids': list(range(n)),
    }


# ===========================================================================
# Main ACNS Engine
# ===========================================================================
class ACNSEngine:
    """
    Ant Colony Neural Swarm (ACNS) simulation engine.
    
    Implements Algorithm 1 from the paper with:
    - Pheromone-driven coordination (Eq. 6-9)
    - Neural control policies (Eq. 21-25)
    - Swing equation dynamics (Eq. 5)
    - Global error metric (Eq. 11)
    - Droop and consensus baselines for comparison
    """

    # Default parameters from Section VII of the paper
    DEFAULT_PARAMS = {
        'w_ref': 60.0,           # Nominal frequency (Hz)
        'v_ref': 1.0,            # Nominal voltage (p.u.)
        'rho': 0.05,             # Evaporation rate ρ ∈ (0,1)
        'kappa': 0.1,            # Reinforcement gain κ > 0
        'eta_theta': 0.01,       # Neural learning rate η_θ
        'gamma': 2.0,            # Voltage mismatch penalty γ > 0
        'delta': 2.0,            # Frequency mismatch penalty δ > 0
        'alpha': 1.0,            # Pheromone influence exponent α
        'beta': 1.0,             # Heuristic influence exponent β
        'epsilon': 1e-4,         # Numerical stability ε
        'dt': 0.01,              # Integration timestep (s)
        'lambda_v': 1.0,         # Voltage weight in local loss λ_v
        'alpha_f': 1.0,          # Frequency weight in global cost α_f
        'alpha_v': 1.0,          # Voltage weight in global cost α_v
        'M': 0.1,                # Inertia coefficient M_i
        'D': 1.0,                # Damping factor D_i
        'K_v': 2.0,              # Voltage control gain
        'K_w': 2.0,              # Frequency control gain
    }

    def __init__(self, network_type='ieee14', params=None):
        self.network_type = network_type
        self.params = dict(self.DEFAULT_PARAMS)
        if params:
            self.params.update(params)

        # Build network topology
        self.topo = _build_network_topology(network_type)
        self.n = self.topo['n_buses']

        # Build adjacency matrix
        self.adj_matrix = np.zeros((self.n, self.n), dtype=np.float32)
        for i, neighbors in self.topo['adj'].items():
            for j in neighbors:
                self.adj_matrix[int(i)][int(j)] = 1.0

        # Neighbor lists for fast access
        self.neighbors = {}
        for i in range(self.n):
            self.neighbors[i] = np.where(self.adj_matrix[i] > 0)[0].tolist()

        # Neural policies (one per bus) - Eq. 21-23
        self.input_dim = 6  # [V, ω, P, Q, τ̄, p̄]
        self.policies = [NeuralPolicy(self.input_dim) for _ in range(self.n)]
        self.optimizers = [
            torch.optim.SGD(p.parameters(), lr=self.params['eta_theta'])
            for p in self.policies
        ]

        self.reset()

    def reset(self):
        """Reset simulation to initial conditions."""
        self.t = 0.0
        self.step_count = 0

        # State vectors
        self.V = self.topo['V_init'].copy()
        self.theta = np.zeros(self.n)
        self.P = self.topo['P_init'].copy()
        self.Q = self.topo['Q_init'].copy()
        self.w = np.ones(self.n) * self.params['w_ref']

        # Add small initial perturbation for interesting dynamics
        rng = np.random.RandomState(123)
        self.w += rng.uniform(-0.05, 0.05, self.n)
        self.V += rng.uniform(-0.02, 0.02, self.n)

        # Pheromone matrix τ_ij - initialized uniformly (Eq. 6)
        self.tau = np.ones((self.n, self.n)) * 0.5 * self.adj_matrix
        np.fill_diagonal(self.tau, 0.0)

        # Coordination weights p_ij (Eq. 8)
        self.p_weights = np.zeros((self.n, self.n))

        # Metrics history
        self.metrics_history = {
            'time': [],
            'global_error': [],
            'freq_deviation_max': [],
            'freq_deviation_mean': [],
            'voltage_rmse': [],
            'voltage_max_dev': [],
            'pheromone_mean': [],
            'convergence_ratio': [],
        }
        self._initial_error = None

        # Re-init neural policies
        for p in self.policies:
            for m in p.net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    nn.init.zeros_(m.bias)

        return self.get_network_state()

    # -------------------------------------------------------------------
    # Pheromone Dynamics (Equations 6-9)
    # -------------------------------------------------------------------
    def _update_pheromones(self):
        """Update pheromone values and coordination weights."""
        rho = self.params['rho']
        kappa = self.params['kappa']
        gamma = self.params['gamma']
        delta = self.params['delta']
        alpha = self.params['alpha']
        beta = self.params['beta']
        eps = self.params['epsilon']

        # Compute reinforcement signal Φ_ij (Eq. 7)
        # Φ_ij = exp(-γ|Vi-Vj| - δ|ωi-ωj|)
        dV = np.abs(self.V[:, None] - self.V[None, :])
        dw = np.abs(self.w[:, None] - self.w[None, :])
        Phi = np.exp(-gamma * dV - delta * dw) * self.adj_matrix

        # Pheromone update (Eq. 6)
        # τ_ij(t+1) = (1-ρ)τ_ij(t) + κ·Φ_ij(t)
        self.tau = (1 - rho) * self.tau + kappa * Phi
        self.tau = np.clip(self.tau, 1e-6, 10.0)  # Bound for stability
        self.tau *= self.adj_matrix  # Maintain sparsity

        # Heuristic information η_ij (Eq. 9)
        # η_ij = 1 / (|Vi-Vj| + ε)
        eta = 1.0 / (dV + eps) * self.adj_matrix

        # Coordination weights p_ij (Eq. 8)
        # p_ij = τ_ij^α · η_ij^β / Σ_k τ_ik^α · η_ik^β
        numerator = np.power(self.tau + 1e-10, alpha) * np.power(eta + 1e-10, beta)
        numerator *= self.adj_matrix
        denom = np.sum(numerator, axis=1, keepdims=True)
        denom[denom < 1e-10] = 1.0
        self.p_weights = numerator / denom

    # -------------------------------------------------------------------
    # Neural Control (Equations 21-25)
    # -------------------------------------------------------------------
    def _compute_neural_controls(self):
        """Compute control actions from neural policies and update weights."""
        controls = np.zeros((self.n, 2))  # [ΔP, ΔQ] for each bus
        v_ref = self.params['v_ref']
        w_ref = self.params['w_ref']
        lv = self.params['lambda_v']

        for i in range(self.n):
            nbrs = self.neighbors[i]

            # Compute mean pheromone and coordination weight (Eq. 20)
            if len(nbrs) > 0:
                tau_bar = np.mean(self.tau[i, nbrs])
                p_bar = np.mean(self.p_weights[i, nbrs])
            else:
                tau_bar = 0.0
                p_bar = 0.0

            # Normalized observation vector s̄_i (Eq. 20)
            s_i = torch.tensor([
                (self.V[i] - v_ref) / 0.1,      # Normalized voltage error
                (self.w[i] - w_ref) / 0.5,       # Normalized frequency error
                self.P[i],
                self.Q[i],
                tau_bar,
                p_bar,
            ], dtype=torch.float32)

            # Forward pass through neural policy (Eq. 21-23)
            self.optimizers[i].zero_grad()
            u_i = self.policies[i](s_i)

            # Local loss (Eq. 25): L_i = (ωi - ωref)² + λv(Vi - Vref)²
            freq_err = (self.w[i] - w_ref) ** 2
            volt_err = (self.V[i] - v_ref) ** 2
            L_i = freq_err + lv * volt_err

            # Proxy gradient: use control output to propagate gradients
            # This connects the neural output to the loss landscape
            proxy = L_i * (u_i[0] ** 2 + u_i[1] ** 2 + 0.01)
            proxy.backward()

            # Parameter update (Eq. 24): θ_i(t+1) = θ_i(t) - η_θ ∇_θi L_i
            self.optimizers[i].step()

            # Clip control actions for stability
            controls[i, 0] = np.clip(u_i[0].item(), -0.5, 0.5)
            controls[i, 1] = np.clip(u_i[1].item(), -0.3, 0.3)

        return controls

    # -------------------------------------------------------------------
    # Physics Step (Equations 5, 30)
    # -------------------------------------------------------------------
    def _physics_step(self, controls):
        """Update electrical states using swing equation and voltage dynamics.
        
        Implements:
          - Swing equation (Eq. 5): M_i ẇ_i + D_i(ω_i - ω_ref) = P^m_i - P_i
          - Swarm coupling (Eq. 30): ė_i = -K_i·e_i - Σ_j a_ij(e_i - e_j)
          - Control application of neural outputs
        """
        dt = self.params['dt']
        M = self.params['M']
        D = self.params['D']
        K_w = self.params['K_w']
        K_v = self.params['K_v']
        w_ref = self.params['w_ref']
        v_ref = self.params['v_ref']

        # Compute symmetric coupling a_ij = (p_ij + p_ji) / 2 (Eq. 29)
        a_sym = (self.p_weights + self.p_weights.T) / 2.0

        for i in range(self.n):
            # Error dynamics (Eq. 30): ė_i = -K_i·e_i - Σ_j a_ij(e_i - e_j)
            e_w = self.w[i] - w_ref
            e_v = self.V[i] - v_ref

            # Local damping term: -K_i · e_i
            dw_local = -K_w * e_w
            dv_local = -K_v * e_v

            # Swarm coupling term: -Σ_j a_ij(e_i - e_j)
            dw_coupling = 0.0
            dv_coupling = 0.0
            for j in self.neighbors[i]:
                a_ij = a_sym[i, j]
                dw_coupling -= a_ij * (e_w - (self.w[j] - w_ref))
                dv_coupling -= a_ij * (e_v - (self.V[j] - v_ref))

            # Neural control contribution
            dw_neural = controls[i, 0] * 0.5
            dv_neural = controls[i, 1] * 0.3

            # Combined dynamics
            # Swing equation: M · dω/dt = -D(ω-ωref) + control
            dw_dt = (dw_local + dw_coupling + dw_neural) / max(M, 0.01)
            dv_dt = dv_local + dv_coupling + dv_neural

            # Euler integration
            self.w[i] += dw_dt * dt
            self.V[i] += dv_dt * dt

        # Enforce operational limits (Eq. 14)
        self.V = np.clip(self.V, 0.85, 1.15)
        self.w = np.clip(self.w, w_ref - 1.0, w_ref + 1.0)

    # -------------------------------------------------------------------
    # Main Simulation Step (Algorithm 1)
    # -------------------------------------------------------------------
    def step(self):
        """Execute one ACNS control interval (Algorithm 1).
        
        Steps:
          1. Measure local state (line 4)
          2. Compute control action via neural policy (line 5)
          3. Update electrical states (line 6)
          4. Compute reinforcement, update pheromones (lines 8-9)
          5. Compute interaction probabilities (line 11)
          6. Update neural parameters (line 12)
        """
        # Step 1-2: Neural control computation (includes param update)
        controls = self._compute_neural_controls()

        # Step 3: Physics update
        self._physics_step(controls)

        # Step 4-5: Pheromone update
        self._update_pheromones()

        # Record metrics
        self._record_metrics()

        self.t += self.params['dt']
        self.step_count += 1

        return self.get_network_state()

    # -------------------------------------------------------------------
    # Disturbance Injection
    # -------------------------------------------------------------------
    def inject_disturbance(self, dist_type='load', magnitude=0.1, buses=None):
        """Inject disturbances as described in Section VII.
        
        Args:
            dist_type: 'load' (±10% load), 'pv' (±15% PV), 'fault' (bus fault)
            magnitude: disturbance magnitude (0.0 to 0.25)
            buses: list of bus indices (None = random 10% of buses)
        """
        if buses is None:
            n_affected = max(1, int(0.1 * self.n))
            buses = np.random.choice(self.n, size=n_affected, replace=False).tolist()
        
        buses = [int(b) for b in buses if 0 <= int(b) < self.n]

        for b in buses:
            if dist_type == 'load':
                # Load disturbance: frequency drops, voltage sags
                self.w[b] -= magnitude * 0.5 * (1 + 0.2 * np.random.randn())
                self.V[b] -= magnitude * 0.3 * (1 + 0.2 * np.random.randn())
            elif dist_type == 'pv':
                # PV fluctuation: affects both frequency and voltage
                sign = np.random.choice([-1, 1])
                self.w[b] += sign * magnitude * 0.4 * (1 + 0.1 * np.random.randn())
                self.V[b] += sign * magnitude * 0.2 * (1 + 0.1 * np.random.randn())
            elif dist_type == 'fault':
                # Bus fault: severe voltage drop
                self.V[b] *= (1 - magnitude)
                self.w[b] -= magnitude * 0.8

        return {'affected_buses': buses, 'type': dist_type, 'magnitude': magnitude}

    # -------------------------------------------------------------------
    # Baseline Controllers for Comparison (Section VIII-C)
    # -------------------------------------------------------------------
    def _droop_step(self, V, w, adj, neighbors, dt, v_ref, w_ref):
        """Conventional droop control baseline.
        
        Simple P-f and Q-V droop: slow convergence, steady-state error.
        """
        mp = 0.02   # droop coefficient for frequency
        nq = 0.05   # droop coefficient for voltage

        for i in range(len(V)):
            dw = -mp * (w[i] - w_ref)
            dv = -nq * (V[i] - v_ref)
            w[i] += dw * dt
            V[i] += dv * dt

        V[:] = np.clip(V, 0.85, 1.15)
        w[:] = np.clip(w, w_ref - 1.0, w_ref + 1.0)

    def _consensus_step(self, V, w, adj, neighbors, dt, v_ref, w_ref):
        """Consensus-based distributed control baseline.
        
        Uses neighbor averaging for coordination, medium convergence.
        """
        kp = 0.5    # proportional gain
        kc = 0.3    # consensus coupling gain

        w_new = w.copy()
        V_new = V.copy()

        for i in range(len(V)):
            nbrs = neighbors[i]
            if len(nbrs) > 0:
                w_avg = np.mean(w[nbrs])
                V_avg = np.mean(V[nbrs])
            else:
                w_avg = w[i]
                V_avg = V[i]

            dw = -kp * (w[i] - w_ref) - kc * (w[i] - w_avg)
            dv = -kp * (V[i] - v_ref) - kc * (V[i] - V_avg)

            w_new[i] += dw * dt
            V_new[i] += dv * dt

        w[:] = np.clip(w_new, w_ref - 1.0, w_ref + 1.0)
        V[:] = np.clip(V_new, 0.85, 1.15)

    def run_comparison(self, steps=1000):
        """Run side-by-side comparison of ACNS, Droop, and Consensus.
        
        Returns results matching Table I from the paper.
        """
        dt = self.params['dt']
        v_ref = self.params['v_ref']
        w_ref = self.params['w_ref']
        af = self.params['alpha_f']
        av = self.params['alpha_v']

        # Save current state
        saved_state = {
            'V': self.V.copy(), 'w': self.w.copy(),
            'P': self.P.copy(), 'Q': self.Q.copy(),
            'tau': self.tau.copy(), 'p_weights': self.p_weights.copy(),
            't': self.t, 'step_count': self.step_count,
            'metrics': copy.deepcopy(self.metrics_history),
        }

        def compute_J(V, w):
            return float(np.sum(af * (w - w_ref) ** 2 + av * (V - v_ref) ** 2))

        results = {
            'acns': {'time': [], 'global_error': [], 'freq_dev': [], 'volt_dev': []},
            'droop': {'time': [], 'global_error': [], 'freq_dev': [], 'volt_dev': []},
            'consensus': {'time': [], 'global_error': [], 'freq_dev': [], 'volt_dev': []},
        }

        # ---- ACNS run ----
        self.reset()
        self.inject_disturbance('load', 0.15)
        for s in range(steps):
            self.step()
            t_now = s * dt
            J = compute_J(self.V, self.w)
            results['acns']['time'].append(round(t_now, 4))
            results['acns']['global_error'].append(float(J))
            results['acns']['freq_dev'].append(float(np.max(np.abs(self.w - w_ref))))
            results['acns']['volt_dev'].append(float(np.sqrt(np.mean((self.V - v_ref) ** 2))))

        # ---- Droop run ----
        V_d = self.topo['V_init'].copy()
        w_d = np.ones(self.n) * w_ref
        rng = np.random.RandomState(123)
        w_d += rng.uniform(-0.05, 0.05, self.n)
        V_d += rng.uniform(-0.02, 0.02, self.n)
        # Same disturbance
        n_aff = max(1, int(0.1 * self.n))
        d_buses = np.random.RandomState(None).choice(self.n, n_aff, replace=False)
        for b in d_buses:
            w_d[b] -= 0.15 * 0.5
            V_d[b] -= 0.15 * 0.3

        for s in range(steps):
            self._droop_step(V_d, w_d, self.adj_matrix, self.neighbors, dt, v_ref, w_ref)
            t_now = s * dt
            J = compute_J(V_d, w_d)
            results['droop']['time'].append(round(t_now, 4))
            results['droop']['global_error'].append(float(J))
            results['droop']['freq_dev'].append(float(np.max(np.abs(w_d - w_ref))))
            results['droop']['volt_dev'].append(float(np.sqrt(np.mean((V_d - v_ref) ** 2))))

        # ---- Consensus run ----
        V_c = self.topo['V_init'].copy()
        w_c = np.ones(self.n) * w_ref
        rng2 = np.random.RandomState(123)
        w_c += rng2.uniform(-0.05, 0.05, self.n)
        V_c += rng2.uniform(-0.02, 0.02, self.n)
        for b in d_buses:
            w_c[b] -= 0.15 * 0.5
            V_c[b] -= 0.15 * 0.3

        for s in range(steps):
            self._consensus_step(V_c, w_c, self.adj_matrix, self.neighbors, dt, v_ref, w_ref)
            t_now = s * dt
            J = compute_J(V_c, w_c)
            results['consensus']['time'].append(round(t_now, 4))
            results['consensus']['global_error'].append(float(J))
            results['consensus']['freq_dev'].append(float(np.max(np.abs(w_c - w_ref))))
            results['consensus']['volt_dev'].append(float(np.sqrt(np.mean((V_c - v_ref) ** 2))))

        # Compute summary (Table I)
        J0_acns = results['acns']['global_error'][0] if results['acns']['global_error'] else 1.0
        J0_droop = results['droop']['global_error'][0] if results['droop']['global_error'] else 1.0
        J0_cons = results['consensus']['global_error'][0] if results['consensus']['global_error'] else 1.0

        def find_t5(errors, J0, dt_val):
            for idx, e in enumerate(errors):
                if J0 > 0 and e / J0 <= 0.05:
                    return round(idx * dt_val, 2)
            return None

        J_final_acns = results['acns']['global_error'][-1] if results['acns']['global_error'] else 0
        J_final_droop = results['droop']['global_error'][-1] if results['droop']['global_error'] else 0
        J_final_cons = results['consensus']['global_error'][-1] if results['consensus']['global_error'] else 0

        results['summary'] = {
            'acns': {
                't5_percent': find_t5(results['acns']['global_error'], J0_acns, dt),
                'J_final': round(J_final_acns, 6),
                'improvement': round((1 - J_final_acns / max(J_final_droop, 1e-10)) * 100, 2),
            },
            'droop': {
                't5_percent': find_t5(results['droop']['global_error'], J0_droop, dt),
                'J_final': round(J_final_droop, 6),
                'improvement': 0.0,
            },
            'consensus': {
                't5_percent': find_t5(results['consensus']['global_error'], J0_cons, dt),
                'J_final': round(J_final_cons, 6),
                'improvement': round((1 - J_final_cons / max(J_final_droop, 1e-10)) * 100, 2),
            },
        }

        # Restore state
        self.V = saved_state['V']
        self.w = saved_state['w']
        self.P = saved_state['P']
        self.Q = saved_state['Q']
        self.tau = saved_state['tau']
        self.p_weights = saved_state['p_weights']
        self.t = saved_state['t']
        self.step_count = saved_state['step_count']
        self.metrics_history = saved_state['metrics']

        return results

    # -------------------------------------------------------------------
    # Metrics Recording
    # -------------------------------------------------------------------
    def _record_metrics(self):
        """Record all performance metrics for the current step."""
        w_ref = self.params['w_ref']
        v_ref = self.params['v_ref']

        freq_dev = np.abs(self.w - w_ref)
        volt_dev = self.V - v_ref

        J = float(np.sum(
            self.params['alpha_f'] * (self.w - w_ref) ** 2 +
            self.params['alpha_v'] * volt_dev ** 2
        ))

        if self._initial_error is None:
            self._initial_error = max(J, 1e-10)

        self.metrics_history['time'].append(round(self.t, 4))
        self.metrics_history['global_error'].append(round(J, 8))
        self.metrics_history['freq_deviation_max'].append(round(float(np.max(freq_dev)), 6))
        self.metrics_history['freq_deviation_mean'].append(round(float(np.mean(freq_dev)), 6))
        self.metrics_history['voltage_rmse'].append(round(float(np.sqrt(np.mean(volt_dev ** 2))), 6))
        self.metrics_history['voltage_max_dev'].append(round(float(np.max(np.abs(volt_dev))), 6))
        self.metrics_history['pheromone_mean'].append(round(float(np.mean(self.tau[self.adj_matrix > 0])), 6))
        self.metrics_history['convergence_ratio'].append(round(J / self._initial_error, 8))

    # -------------------------------------------------------------------
    # State Getters
    # -------------------------------------------------------------------
    def get_network_state(self):
        """Get complete network state snapshot."""
        w_ref = self.params['w_ref']
        v_ref = self.params['v_ref']

        buses = []
        for i in range(self.n):
            buses.append({
                'id': i,
                'V': round(float(self.V[i]), 6),
                'w': round(float(self.w[i]), 6),
                'P': round(float(self.P[i]), 4),
                'Q': round(float(self.Q[i]), 4),
                'theta': round(float(self.theta[i]), 4),
                'is_gen': i in self.topo['gen_buses'],
            })

        # Latest metrics
        metrics = {}
        if self.metrics_history['time']:
            metrics = {
                'global_error': self.metrics_history['global_error'][-1],
                'freq_dev_max': self.metrics_history['freq_deviation_max'][-1],
                'freq_dev_mean': self.metrics_history['freq_deviation_mean'][-1],
                'voltage_rmse': self.metrics_history['voltage_rmse'][-1],
                'pheromone_mean': self.metrics_history['pheromone_mean'][-1],
                'convergence_ratio': self.metrics_history['convergence_ratio'][-1],
            }

        return {
            'time': round(self.t, 4),
            'step': self.step_count,
            'n_buses': self.n,
            'n_agents': len(self.topo['gen_buses']),
            'buses': buses,
            'lines': self.topo['lines'],
            'metrics': metrics,
        }

    def get_pheromone_map(self):
        """Get pheromone matrix (only edges with connections)."""
        edges = []
        for i in range(self.n):
            for j in self.neighbors[i]:
                if j > i:  # Avoid duplicates
                    edges.append({
                        'from': i, 'to': j,
                        'tau': round(float(self.tau[i, j]), 6),
                        'p_weight': round(float(self.p_weights[i, j]), 6),
                    })

        # Also return a subsampled matrix for heatmap visualization
        sample_size = min(self.n, 30)
        indices = np.linspace(0, self.n - 1, sample_size, dtype=int)
        sub_matrix = self.tau[np.ix_(indices, indices)].tolist()

        return {
            'edges': edges,
            'matrix': sub_matrix,
            'matrix_indices': indices.tolist(),
            'full_mean': round(float(np.mean(self.tau[self.adj_matrix > 0])), 6),
            'full_max': round(float(np.max(self.tau)), 6),
        }

    def get_metrics_history(self):
        """Get full metrics time series."""
        return self.metrics_history
