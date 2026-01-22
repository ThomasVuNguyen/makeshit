#!/usr/bin/env python3
"""
MuJoCo Training Script for Makeshit V2

Reads the exported model from ../output/ and runs training with visualization.
Serves a live view on port 1306 + opens MuJoCo viewer window.

Usage:
    cd train
    source venv/bin/activate
    python train.py
"""

import os
import sys
import time
import threading
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
from flask import Flask, jsonify
from flask_cors import CORS

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "output"
MODEL_XML = OUTPUT_DIR / "model.xml"
PORT = 1306

# Flask app
app = Flask(__name__)
CORS(app)

# Global state
sim_state = {
    "running": False,
    "time": 0.0,
    "reward": 0.0,
    "episode": 0,
    "step": 0,
}

model = None
data = None


def load_model():
    """Load the MuJoCo model."""
    global model, data
    
    if not MODEL_XML.exists():
        print(f"❌ Model not found at {MODEL_XML}")
        return False
    
    print(f"📂 Loading model from {MODEL_XML}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    print(f"✅ Model loaded: {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators")
    return True


# Flask routes
@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MuJoCo Training</title>
        <style>
            body { font-family: system-ui; background: #1a1a2e; color: white; padding: 40px; }
            h1 { color: #4a90d9; }
            .stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; max-width: 600px; }
            .stat { background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px; }
            .stat-value { font-size: 2em; font-weight: bold; color: #4a90d9; }
            .stat-label { opacity: 0.7; margin-top: 8px; }
            .status { display: inline-block; padding: 8px 16px; border-radius: 20px; margin-bottom: 20px; }
            .status.running { background: #22c55e; }
            .status.stopped { background: #ef4444; }
        </style>
    </head>
    <body>
        <h1>🔬 MuJoCo Training</h1>
        <div id="status" class="status">Loading...</div>
        <div class="stats">
            <div class="stat"><div class="stat-value" id="time">0.0s</div><div class="stat-label">Simulation Time</div></div>
            <div class="stat"><div class="stat-value" id="episode">0</div><div class="stat-label">Episode</div></div>
            <div class="stat"><div class="stat-value" id="step">0</div><div class="stat-label">Steps</div></div>
            <div class="stat"><div class="stat-value" id="reward">0.0</div><div class="stat-label">Episode Reward</div></div>
        </div>
        <p style="margin-top: 30px; opacity: 0.6;">MuJoCo viewer window runs on main thread.</p>
        <script>
            async function update() {
                try {
                    const res = await fetch('/status');
                    const d = await res.json();
                    document.getElementById('time').textContent = d.time.toFixed(2) + 's';
                    document.getElementById('episode').textContent = d.episode;
                    document.getElementById('step').textContent = d.step;
                    document.getElementById('reward').textContent = d.reward.toFixed(2);
                    const s = document.getElementById('status');
                    s.textContent = d.running ? '🟢 Running' : '🔴 Stopped';
                    s.className = 'status ' + (d.running ? 'running' : 'stopped');
                } catch (e) { console.error(e); }
            }
            setInterval(update, 100);
            update();
        </script>
    </body>
    </html>
    """


@app.route("/status")
def status():
    return jsonify(sim_state)


def run_flask():
    """Run Flask in background thread."""
    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)


def main():
    print("=" * 50)
    print("  Makeshit V2 - MuJoCo Training")
    print("=" * 50)
    
    if not load_model():
        print("\n💡 Export model from web app first.")
        sys.exit(1)
    
    # Start Flask in background
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print(f"\n🌐 Dashboard: http://localhost:{PORT}")
    
    # Run viewer on MAIN thread (required for macOS)
    print("🎬 Opening MuJoCo viewer...")
    sim_state["running"] = True
    
    episode = 0
    step = 0
    episode_reward = 0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("✅ Viewer opened! Horn should be oscillating.")
        
        while viewer.is_running():
            # Apply sinusoidal motor control
            if model.nu > 0:
                data.ctrl[0] = np.sin(data.time * 2) * 0.8
            
            # Step physics
            mujoco.mj_step(model, data)
            step += 1
            
            # Simple reward
            episode_reward += max(0, 1.0 - abs(data.qpos[0]))
            
            # Update dashboard state
            sim_state["time"] = data.time
            sim_state["step"] = step
            sim_state["episode"] = episode
            sim_state["reward"] = episode_reward
            
            # Episode boundary
            if step % 1000 == 0:
                print(f"Episode {episode}: {step} steps, reward={episode_reward:.1f}, angle={np.degrees(data.qpos[0]):.1f}°")
                episode += 1
                episode_reward = 0
            
            # Sync viewer
            viewer.sync()
            
            # Real-time pacing
            time.sleep(model.opt.timestep)
    
    sim_state["running"] = False
    print("\n⏹ Viewer closed. Simulation stopped.")


if __name__ == "__main__":
    main()
