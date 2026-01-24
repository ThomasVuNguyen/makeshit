"""
Bipedal Walking Robot Training (10 Parallel Environments)
PPO training with web-based visualization on port 1306
"""

import threading
import time
import io
from pathlib import Path

import numpy as np
from flask import Flask, Response, render_template_string, jsonify
from flask_cors import CORS
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import imageio

# Import our custom environment
from env import BipedalWalkerEnv


# Configuration
NUM_ENVS = 10  # Number of parallel environments


# Global variables for frame sharing
current_frame = None
frame_lock = threading.Lock()
training_stats = {
    "episode": 0, 
    "reward": 0.0, 
    "steps": 0, 
    "best_reward": -float('inf'),
    "num_envs": NUM_ENVS
}


# Flask app for visualization
app = Flask(__name__)
CORS(app)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Bipedal Walker Training (10 Robots) - Port 1306</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
            color: #fff;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #f093fb, #f5576c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #a8a8b3; margin-bottom: 30px; font-size: 1.1rem; }
        .parallel-badge {
            display: inline-block; padding: 5px 15px;
            background: linear-gradient(90deg, #00d4ff, #0099ff);
            border-radius: 15px; font-weight: bold; font-size: 0.9rem; margin-left: 10px;
        }
        .container { display: flex; gap: 30px; flex-wrap: wrap; justify-content: center; }
        .video-container {
            background: rgba(255, 255, 255, 0.05); border-radius: 20px; padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.1);
        }
        #video-feed { border-radius: 12px; width: 640px; height: 480px; background: #000; }
        .stats-panel {
            background: rgba(255, 255, 255, 0.05); border-radius: 20px; padding: 30px; min-width: 280px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .stat { margin-bottom: 25px; }
        .stat-label { font-size: 0.9rem; color: #a8a8b3; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; }
        .stat-value { font-size: 2.2rem; font-weight: bold; background: linear-gradient(90deg, #f093fb, #f5576c); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .status-badge { display: inline-block; padding: 8px 20px; background: linear-gradient(90deg, #f093fb, #f5576c); border-radius: 20px; font-weight: bold; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        .parallel-info { margin-top: 20px; padding: 15px; background: rgba(0, 212, 255, 0.1); border-radius: 10px; border: 1px solid rgba(0, 212, 255, 0.3); }
        .parallel-info h3 { color: #00d4ff; margin-bottom: 10px; font-size: 1rem; }
        .parallel-info p { color: #a8a8b3; font-size: 0.85rem; line-height: 1.5; }
    </style>
</head>
<body>
    <h1>🦿 Bipedal Walking Robot <span class="parallel-badge">×10 PARALLEL</span></h1>
    <p class="subtitle">Learning to walk with Reinforcement Learning (PPO)</p>
    <div class="container">
        <div class="video-container">
            <img id="video-feed" src="/video_feed" alt="Training visualization">
        </div>
        <div class="stats-panel">
            <div class="stat"><div class="stat-label">Status</div><div class="status-badge">🏃 TRAINING (10 envs)</div></div>
            <div class="stat"><div class="stat-label">Episodes (all envs)</div><div class="stat-value" id="episode">0</div></div>
            <div class="stat"><div class="stat-label">Avg Reward</div><div class="stat-value" id="reward">0.00</div></div>
            <div class="stat"><div class="stat-label">Best Reward</div><div class="stat-value" id="best_reward">-∞</div></div>
            <div class="stat"><div class="stat-label">Total Steps</div><div class="stat-value" id="steps">0</div></div>
            <div class="parallel-info">
                <h3>⚡ Parallel Training</h3>
                <p>Running <strong>10 robots</strong> simultaneously for 10× faster learning!</p>
            </div>
        </div>
    </div>
    <script>
        async function updateStats() {
            try {
                const response = await fetch('/stats');
                const data = await response.json();
                document.getElementById('episode').textContent = data.episode;
                document.getElementById('reward').textContent = data.reward.toFixed(2);
                document.getElementById('best_reward').textContent = data.best_reward.toFixed(2);
                document.getElementById('steps').textContent = data.steps;
            } catch (e) {}
        }
        setInterval(updateStats, 500);
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/stats')
def stats():
    return jsonify({
        "episode": int(training_stats["episode"]),
        "reward": float(training_stats["reward"]),
        "best_reward": float(training_stats["best_reward"]),
        "steps": int(training_stats["steps"]),
        "num_envs": int(training_stats["num_envs"])
    })


def generate_frames():
    """Generator for MJPEG streaming."""
    while True:
        with frame_lock:
            if current_frame is not None:
                try:
                    buffer = io.BytesIO()
                    imageio.imwrite(buffer, current_frame, format='jpeg')
                    frame_bytes = buffer.getvalue()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    pass
        time.sleep(1/30)


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def run_server():
    """Run Flask server in a separate thread."""
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # Suppress HTTP request logs
    app.run(host='0.0.0.0', port=1306, threaded=True, use_reloader=False)


def make_env(render_mode=None):
    """Factory function for creating environments."""
    def _init():
        return BipedalWalkerEnv(render_mode=render_mode)
    return _init


class RenderingCallback(BaseCallback):
    """Callback that renders one of the training environments."""
    
    def __init__(self, render_env, model_ref, verbose=0):
        super().__init__(verbose)
        self.render_env = render_env
        self.model_ref = model_ref  # List to get model after creation
        self.episode_rewards = []
        self.episode_count = 0
        self.current_rewards = np.zeros(NUM_ENVS)
        self.render_obs = None
        self.render_step = 0
        
    def _on_training_start(self):
        """Initialize render environment when training starts."""
        self.render_obs, _ = self.render_env.reset()
        
    def _on_step(self) -> bool:
        global current_frame, training_stats
        
        # Update render environment with learned policy every few steps
        if self.num_timesteps % 5 == 0 and self.model_ref[0] is not None:
            try:
                action, _ = self.model_ref[0].predict(self.render_obs, deterministic=True)
                self.render_obs, _, terminated, truncated, _ = self.render_env.step(action)
                
                if terminated or truncated:
                    self.render_obs, _ = self.render_env.reset()
                
                frame = self.render_env.render()
                if frame is not None:
                    with frame_lock:
                        current_frame = frame.copy()
            except Exception as e:
                pass
        
        # Track rewards from all environments
        rewards = self.locals.get('rewards', np.zeros(NUM_ENVS))
        self.current_rewards += rewards
        training_stats["steps"] = self.num_timesteps
        
        # Check for episode ends
        dones = self.locals.get('dones', np.zeros(NUM_ENVS, dtype=bool))
        for i, done in enumerate(dones):
            if done:
                self.episode_count += 1
                self.episode_rewards.append(self.current_rewards[i])
                
                if self.current_rewards[i] > training_stats["best_reward"]:
                    training_stats["best_reward"] = self.current_rewards[i]
                
                self.current_rewards[i] = 0
        
        training_stats["episode"] = self.episode_count
        
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-100:]
            training_stats["reward"] = np.mean(recent_rewards)
        
        if self.episode_count > 0 and self.episode_count % 100 == 0 and dones.any():
            avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            print(f"📊 Episodes: {self.episode_count} | Avg Reward: {avg_reward:.2f} | Best: {training_stats['best_reward']:.2f} | Steps: {self.num_timesteps}")
        
        return True


def train():
    """Main training loop with 10 parallel environments."""
    global current_frame, training_stats
    
    print(f"🦿 Starting Bipedal Walking Robot Training")
    print(f"⚡ Running {NUM_ENVS} parallel environments for faster learning!")
    print("📺 Visualization available at http://localhost:1306")
    print("-" * 50)
    
    # Create vectorized environment with parallel instances
    vec_env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])
    vec_env = VecMonitor(vec_env)
    
    # Render environment in main process
    render_env = BipedalWalkerEnv(render_mode="rgb_array")
    
    # Initial render
    render_env.reset()
    frame = render_env.render()
    if frame is not None:
        with frame_lock:
            current_frame = frame.copy()
    
    # Model reference holder (list so callback can access after creation)
    model_ref = [None]
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )
    )
    model_ref[0] = model
    
    # Create callback
    callback = RenderingCallback(render_env, model_ref)
    
    print(f"\n🎯 Training for 1,000,000 timesteps...")
    print(f"🤖 Robots: {NUM_ENVS} bipedal walkers training in parallel")
    print("🎓 Goal: Learn to walk forward\n")
    
    model.learn(
        total_timesteps=1000000,
        callback=callback,
        progress_bar=False
    )
    
    model.save("bipedal_walker_ppo_parallel")
    print("\n✅ Training complete! Model saved to bipedal_walker_ppo_parallel.zip")
    
    vec_env.close()
    render_env.close()


def main():
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    time.sleep(1)
    print("=" * 50)
    print("🌐 Web server started on http://localhost:1306")
    print("=" * 50)
    
    train()


if __name__ == "__main__":
    main()
