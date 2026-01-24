"""
Multi-Robot Bipedal Walking Training
10 robots visible simultaneously in one scene!
"""

import threading
import time
import io
from pathlib import Path

import numpy as np
from flask import Flask, Response, render_template_string, jsonify
from flask_cors import CORS
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import imageio

from multi_env import MultiRobotWalkerEnv


NUM_ROBOTS = 10

current_frame = None
frame_lock = threading.Lock()
training_stats = {"episode": 0, "reward": 0.0, "steps": 0, "best_reward": -float('inf')}


app = Flask(__name__)
CORS(app)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>10 Bipedal Robots Training - Port 1306</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh; color: #fff;
            display: flex; flex-direction: column; align-items: center; padding: 20px;
        }
        h1 { font-size: 2.5rem; margin-bottom: 10px;
            background: linear-gradient(90deg, #f093fb, #f5576c);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .subtitle { color: #a8a8b3; margin-bottom: 20px; }
        .container { display: flex; gap: 30px; flex-wrap: wrap; justify-content: center; }
        .video-container {
            background: rgba(255,255,255,0.05); border-radius: 20px; padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1);
        }
        #video-feed { border-radius: 12px; width: 960px; height: 480px; background: #000; }
        .stats-panel {
            background: rgba(255,255,255,0.05); border-radius: 20px; padding: 25px; min-width: 260px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1);
        }
        .stat { margin-bottom: 20px; }
        .stat-label { font-size: 0.85rem; color: #a8a8b3; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 5px; }
        .stat-value { font-size: 2rem; font-weight: bold;
            background: linear-gradient(90deg, #f093fb, #f5576c);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .status-badge {
            display: inline-block; padding: 8px 16px;
            background: linear-gradient(90deg, #00d4ff, #0099ff);
            border-radius: 20px; font-weight: bold; animation: pulse 2s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        .robot-colors { margin-top: 15px; display: flex; flex-wrap: wrap; gap: 5px; }
        .robot-color { width: 20px; height: 20px; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>🦿 10 Bipedal Robots Training</h1>
    <p class="subtitle">All robots learning to walk together!</p>
    <div class="container">
        <div class="video-container">
            <img id="video-feed" src="/video_feed" alt="10 robots training">
        </div>
        <div class="stats-panel">
            <div class="stat"><div class="stat-label">Status</div><div class="status-badge">🏃 10 ROBOTS</div></div>
            <div class="stat"><div class="stat-label">Episode</div><div class="stat-value" id="episode">0</div></div>
            <div class="stat"><div class="stat-label">Total Reward</div><div class="stat-value" id="reward">0</div></div>
            <div class="stat"><div class="stat-label">Best Reward</div><div class="stat-value" id="best_reward">-∞</div></div>
            <div class="stat"><div class="stat-label">Steps</div><div class="stat-value" id="steps">0</div></div>
            <div class="stat">
                <div class="stat-label">Robot Colors</div>
                <div class="robot-colors">
                    <div class="robot-color" style="background:#4d99e6"></div>
                    <div class="robot-color" style="background:#e64d4d"></div>
                    <div class="robot-color" style="background:#4de64d"></div>
                    <div class="robot-color" style="background:#e6e64d"></div>
                    <div class="robot-color" style="background:#e6801a"></div>
                    <div class="robot-color" style="background:#804de6"></div>
                    <div class="robot-color" style="background:#e64de6"></div>
                    <div class="robot-color" style="background:#4de6e6"></div>
                    <div class="robot-color" style="background:#b3b3b3"></div>
                    <div class="robot-color" style="background:#ffcc33"></div>
                </div>
            </div>
        </div>
    </div>
    <script>
        setInterval(async () => {
            try {
                const r = await fetch('/stats');
                const d = await r.json();
                document.getElementById('episode').textContent = d.episode;
                document.getElementById('reward').textContent = d.reward.toFixed(1);
                document.getElementById('best_reward').textContent = d.best_reward.toFixed(1);
                document.getElementById('steps').textContent = d.steps;
            } catch(e) {}
        }, 500);
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
        "steps": int(training_stats["steps"])
    })


def generate_frames():
    while True:
        with frame_lock:
            if current_frame is not None:
                try:
                    buf = io.BytesIO()
                    imageio.imwrite(buf, current_frame, format='jpeg')
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.getvalue() + b'\r\n')
                except: pass
        time.sleep(1/30)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def run_server():
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=1306, threaded=True, use_reloader=False)


class RenderCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.episode_count = 0
        self.episode_reward = 0
        
    def _on_step(self):
        global current_frame, training_stats
        
        # Render every step
        frame = self.env.render()
        if frame is not None:
            with frame_lock:
                current_frame = frame.copy()
        
        self.episode_reward += self.locals.get('rewards', [0])[0]
        training_stats["steps"] = self.num_timesteps
        training_stats["reward"] = self.episode_reward
        
        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.episode_count += 1
            training_stats["episode"] = self.episode_count
            if self.episode_reward > training_stats["best_reward"]:
                training_stats["best_reward"] = self.episode_reward
            if self.episode_count % 10 == 0:
                print(f"📊 Episode {self.episode_count} | Reward: {self.episode_reward:.1f} | Best: {training_stats['best_reward']:.1f}")
            self.episode_reward = 0
        return True


def train():
    global current_frame
    
    print("🦿 Starting 10-Robot Bipedal Walking Training")
    print("📺 View all 10 robots at http://localhost:1306")
    print("-" * 50)
    
    env = MultiRobotWalkerEnv(render_mode="rgb_array")
    vec_env = DummyVecEnv([lambda: env])
    
    # Initial render
    env.reset()
    frame = env.render()
    if frame is not None:
        with frame_lock:
            current_frame = frame.copy()
    
    # PPO with larger network for 60 outputs
    model = PPO(
        "MlpPolicy", vec_env, verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs=dict(net_arch=dict(pi=[512, 512], vf=[512, 512]))
    )
    
    callback = RenderCallback(env)
    
    print(f"\n🎯 Training 10 robots for 500,000 timesteps...")
    print("🤖 60 motors total (6 per robot × 10)")
    print("🎓 Goal: Learn to walk forward together\n")
    
    model.learn(total_timesteps=500000, callback=callback, progress_bar=False)
    
    model.save("multi_robot_walker_ppo")
    print("\n✅ Training complete! Model saved.")
    env.close()


def main():
    threading.Thread(target=run_server, daemon=True).start()
    time.sleep(1)
    print("=" * 50)
    print("🌐 Server at http://localhost:1306")
    print("=" * 50)
    train()


if __name__ == "__main__":
    main()
