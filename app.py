"""
ACNS Web Application - Flask Server
=====================================
REST API + WebSocket server for the ACNS Power Network Controller.

Endpoints:
  GET  /                         - Serve dashboard
  GET  /api/networks             - List available IEEE networks
  POST /api/simulation/create    - Create new simulation
  POST /api/simulation/step      - Advance one step
  POST /api/simulation/run       - Run N steps with WebSocket streaming
  POST /api/simulation/stop      - Stop running simulation
  POST /api/simulation/disturbance - Inject disturbance
  GET  /api/simulation/state     - Get current state
  GET  /api/simulation/metrics   - Get metrics history
  GET  /api/simulation/pheromone - Get pheromone map
  POST /api/simulation/compare   - Run baseline comparison
  POST /api/simulation/reset     - Reset simulation
  GET  /api/simulation/export    - Export results as JSON
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from acns_engine import ACNSEngine
import threading
import time
import json
import traceback
import os

# ---------------------------------------------------------------------------
# Application Setup
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global simulation state
engine = None
simulation_running = False
sim_thread = None
sim_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Static File Serving
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    """Serve the main dashboard."""
    return send_from_directory('static', 'index.html')


# ---------------------------------------------------------------------------
# API: Network Information
# ---------------------------------------------------------------------------
@app.route('/api/networks', methods=['GET'])
def list_networks():
    """List available IEEE benchmark networks."""
    return jsonify({
        'status': 'success',
        'networks': [
            {'id': 'ieee14', 'name': 'IEEE 14-Bus', 'buses': 14, 'description': 'Small test network'},
            {'id': 'ieee30', 'name': 'IEEE 30-Bus', 'buses': 30, 'description': 'Medium test network'},
            {'id': 'ieee118', 'name': 'IEEE 118-Bus', 'buses': 118, 'description': 'Large-scale benchmark (paper)'},
        ]
    })


# ---------------------------------------------------------------------------
# API: Simulation Management
# ---------------------------------------------------------------------------
@app.route('/api/simulation/create', methods=['POST'])
def create_simulation():
    """Create and initialize a new ACNS simulation."""
    global engine
    data = request.json or {}
    network = data.get('network', 'ieee14')
    params = data.get('params', {})

    try:
        with sim_lock:
            engine = ACNSEngine(network_type=network, params=params)
        state = engine.get_network_state()
        return jsonify({
            'status': 'success',
            'message': f'Simulation created for {network} ({state["n_buses"]} buses)',
            'state': state,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/simulation/step', methods=['POST'])
def step_simulation():
    """Advance simulation by one step."""
    if not engine:
        return jsonify({'status': 'error', 'message': 'No simulation initialized'}), 400

    try:
        with sim_lock:
            state = engine.step()
        return jsonify({'status': 'success', 'state': state})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/simulation/run', methods=['POST'])
def run_simulation():
    """Run simulation for N steps with WebSocket streaming."""
    global simulation_running, sim_thread

    if not engine:
        return jsonify({'status': 'error', 'message': 'No simulation initialized'}), 400

    data = request.json or {}
    steps = min(data.get('steps', 500), 10000)
    speed = max(data.get('speed', 1), 1)

    def run_loop():
        global simulation_running
        try:
            for i in range(steps):
                if not simulation_running:
                    break

                with sim_lock:
                    state = engine.step()

                # Emit every `speed` steps for performance
                if i % speed == 0:
                    socketio.emit('state_update', state)
                    socketio.sleep(0.005)  # Small delay for smooth rendering

            socketio.emit('sim_complete', {
                'status': 'done',
                'final_state': engine.get_network_state(),
            })
        except Exception as e:
            socketio.emit('sim_error', {'error': str(e)})
            traceback.print_exc()
        finally:
            simulation_running = False

    if not simulation_running:
        simulation_running = True
        sim_thread = socketio.start_background_task(run_loop)

    return jsonify({'status': 'success', 'message': f'Running {steps} steps (speed={speed}x)'})


@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    """Stop a running simulation."""
    global simulation_running
    simulation_running = False
    return jsonify({'status': 'success', 'message': 'Simulation stopped'})


@app.route('/api/simulation/reset', methods=['POST'])
def reset_simulation():
    """Reset simulation to initial state."""
    if not engine:
        return jsonify({'status': 'error', 'message': 'No simulation initialized'}), 400

    with sim_lock:
        state = engine.reset()
    return jsonify({'status': 'success', 'message': 'Simulation reset', 'state': state})


# ---------------------------------------------------------------------------
# API: Disturbance Injection
# ---------------------------------------------------------------------------
@app.route('/api/simulation/disturbance', methods=['POST'])
def inject_disturbance():
    """Inject a disturbance into the running simulation."""
    if not engine:
        return jsonify({'status': 'error', 'message': 'No simulation initialized'}), 400

    data = request.json or {}
    dist_type = data.get('type', 'load')
    magnitude = float(data.get('magnitude', 0.1))
    buses = data.get('buses', None)

    try:
        with sim_lock:
            result = engine.inject_disturbance(dist_type, magnitude, buses)
        return jsonify({'status': 'success', 'message': f'{dist_type} disturbance injected', 'details': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ---------------------------------------------------------------------------
# API: State and Metrics Queries
# ---------------------------------------------------------------------------
@app.route('/api/simulation/state', methods=['GET'])
def get_state():
    """Get current simulation state."""
    if not engine:
        return jsonify({'status': 'error', 'message': 'No simulation initialized'}), 400
    return jsonify({'status': 'success', 'state': engine.get_network_state()})


@app.route('/api/simulation/metrics', methods=['GET'])
def get_metrics():
    """Get full metrics history."""
    if not engine:
        return jsonify({'status': 'error', 'message': 'No simulation initialized'}), 400
    return jsonify({'status': 'success', 'metrics': engine.get_metrics_history()})


@app.route('/api/simulation/pheromone', methods=['GET'])
def get_pheromone():
    """Get pheromone map data."""
    if not engine:
        return jsonify({'status': 'error', 'message': 'No simulation initialized'}), 400
    return jsonify({'status': 'success', 'pheromone': engine.get_pheromone_map()})


# ---------------------------------------------------------------------------
# API: Comparison Mode
# ---------------------------------------------------------------------------
@app.route('/api/simulation/compare', methods=['POST'])
def run_comparison():
    """Run ACNS vs Droop vs Consensus comparison."""
    if not engine:
        return jsonify({'status': 'error', 'message': 'No simulation initialized'}), 400

    data = request.json or {}
    steps = min(data.get('steps', 1000), 5000)

    try:
        with sim_lock:
            results = engine.run_comparison(steps)
        return jsonify({'status': 'success', 'results': results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ---------------------------------------------------------------------------
# API: Export Results
# ---------------------------------------------------------------------------
@app.route('/api/simulation/export', methods=['GET'])
def export_results():
    """Export complete simulation results as JSON."""
    if not engine:
        return jsonify({'status': 'error', 'message': 'No simulation initialized'}), 400

    data = {
        'network': engine.network_type,
        'parameters': engine.params,
        'metrics': engine.get_metrics_history(),
        'final_state': engine.get_network_state(),
        'pheromone': engine.get_pheromone_map(),
    }
    return jsonify(data)


# ---------------------------------------------------------------------------
# WebSocket Events
# ---------------------------------------------------------------------------
@socketio.on('connect')
def on_connect():
    """Handle WebSocket connection."""
    emit('connected', {'status': 'connected', 'message': 'ACNS server connected'})


@socketio.on('disconnect')
def on_disconnect():
    """Handle WebSocket disconnection."""
    pass


@socketio.on('request_state')
def on_request_state():
    """Handle state request via WebSocket."""
    if engine:
        emit('state_update', engine.get_network_state())


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("  ACNS Power Network Controller")
    print("  Ant Colony Neural Swarm Coordination")
    print("=" * 60)
    print(f"  Dashboard: http://localhost:5000")
    print(f"  API Base:  http://localhost:5000/api/")
    print("=" * 60)

    socketio.run(app, debug=True, host='0.0.0.0', port=5000,
                 allow_unsafe_werkzeug=True)
