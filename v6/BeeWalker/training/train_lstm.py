"""
BeeWalker LSTM Training Script
Uses RecurrentPPO (LSTM-based PPO) for temporal awareness.

Web Dashboard at http://127.0.0.1:1306 shows:
- Training progress plot (auto-updates)
- Latest 5 training videos

Usage:
    python train_lstm.py                # Start fresh
    python train_lstm.py --resume PATH  # Resume from checkpoint
"""
import os
import sys
os.environ['MUJOCO_GL'] = 'egl'

# Add BeeWalker root to path for cross-package imports
_BEEWALKER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BEEWALKER_ROOT)

import threading
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, Response, send_file
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
import numpy as np
import cv2
import imageio
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BEEWALKER_ROOT = Path(_BEEWALKER_ROOT)

from env.bee_walker_env import BeeWalkerEnv
from training.train import ConfigurableRewardEnv, REWARD_CONFIGS


# Use speed config (best performer from previous training)
config = REWARD_CONFIGS["speed"]

# Flask app for dashboard
app = Flask(__name__)
stats = {"steps": 0, "reward": 0, "best": 0, "ep_len": 0, "fps": 0}
run_dir = None  # Set in main()


def make_env(rank, seed=0):
    def _init():
        from gymnasium.wrappers import TimeLimit
        env = ConfigurableRewardEnv(config, render_mode="rgb_array")
        env = TimeLimit(env, max_episode_steps=1000)  # Force episode end for stats
        env = Monitor(env)  # Track episode stats
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


class TrainingCallback(BaseCallback):
    """Callback for video recording, plotting, and reward logging."""
    
    def __init__(self, eval_env, save_dir, video_freq=100_000, plot_freq=10_000):
        super().__init__()
        self.eval_env = eval_env
        self.save_dir = Path(save_dir)
        self.video_freq = video_freq
        self.plot_freq = plot_freq
        self.best_reward = float('-inf')
        self.reward_log = []
        self._fps_timer = time.time()
        self._fps_steps = 0
    
    def _on_step(self):
        stats["steps"] = self.num_timesteps
        
        # Track FPS
        now = time.time()
        self._fps_steps += 1
        elapsed = now - self._fps_timer
        if elapsed >= 2.0:
            stats["fps"] = self._fps_steps / elapsed
            self._fps_steps = 0
            self._fps_timer = now
        
        # Update plot periodically
        if self.num_timesteps % self.plot_freq == 0 and len(self.reward_log) > 2:
            self._update_plot()
        
        # Save video every video_freq steps
        if self.num_timesteps % self.video_freq == 0 and self.num_timesteps > 0:
            self._record_video()
        
        return True
    
    def _on_rollout_end(self):
        if len(self.model.ep_info_buffer) > 0:
            rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            lengths = [ep["l"] for ep in self.model.ep_info_buffer]
            mean_reward = np.mean(rewards)
            mean_length = np.mean(lengths)
            stats["reward"] = mean_reward
            stats["ep_len"] = mean_length
            
            self.reward_log.append({
                "steps": self.num_timesteps,
                "mean_reward": float(mean_reward),
                "mean_length": float(mean_length),
            })
            
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                stats["best"] = mean_reward
                print(f"  ★ New best: {mean_reward:.1f} (ep_len: {mean_length:.0f})")
                self.model.save(str(self.save_dir.parent / "best_model"))
            
            if len(self.reward_log) % 3 == 0:  # Save every 3 rollouts
                self._save_reward_log()
                self._update_plot()
    
    def _update_plot(self):
        """Generate training progress plot."""
        try:
            if len(self.reward_log) < 3:
                return
            
            steps = [d["steps"] for d in self.reward_log]
            rewards = [d["mean_reward"] for d in self.reward_log]
            lengths = [d["mean_length"] for d in self.reward_log]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), facecolor='#1a1a2e')
            
            for ax in [ax1, ax2]:
                ax.set_facecolor('#16213e')
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.2)
                for s in ax.spines.values():
                    s.set_color('#333')
            
            # Reward plot
            ax1.plot(np.array(steps)/1e3, rewards, color='#00ff88', linewidth=2)
            ax1.set_ylabel('Mean Reward', color='white')
            ax1.set_title(f'BeeWalker LSTM Training — {self.num_timesteps:,} steps @ {stats["fps"]:.0f} sps', 
                         color='white', fontweight='bold')
            ax1.axhline(y=max(rewards), color='#ffff44', linestyle='--', alpha=0.5, label=f'Best: {max(rewards):.1f}')
            ax1.legend(facecolor='#16213e', edgecolor='#666', labelcolor='white')
            
            # Episode length plot
            ax2.plot(np.array(steps)/1e3, lengths, color='#ff8800', linewidth=2)
            ax2.set_ylabel('Episode Length', color='white')
            ax2.set_xlabel('Steps (K)', color='white')
            
            plt.tight_layout()
            plt.savefig(self.save_dir.parent / 'progress.png', dpi=100, facecolor='#1a1a2e')
            plt.close()
        except Exception as e:
            print(f"Plot error: {e}")
    
    def _record_video(self):
        """Record evaluation video using deterministic policy."""
        obs = self.eval_env.reset()[0]
        frames = []
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        for _ in range(300):  # 10 seconds at 30fps
            action, lstm_states = self.model.predict(
                obs, state=lstm_states, 
                episode_start=episode_starts,
                deterministic=True
            )
            obs, _, term, trunc, _ = self.eval_env.step(action)
            frame = self.eval_env.render()
            if frame is not None:
                frames.append(frame)
            
            episode_starts = np.array([term or trunc])
            if term or trunc:
                obs = self.eval_env.reset()[0]
        
        if frames:
            video_dir = self.save_dir
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"step_{self.num_timesteps:09d}.mp4"
            imageio.mimwrite(
                str(video_path), frames, fps=30, codec='libx264',
                output_params=['-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '23']
            )
            print(f"  📹 Saved: {video_path.name}")
    
    def _save_reward_log(self):
        log_path = self.save_dir.parent / "reward_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.reward_log, f)


# ============================================================================
# WEB DASHBOARD
# ============================================================================

@app.route('/')
def index():
    return """<!DOCTYPE html>
<html>
<head>
    <title>BeeWalker LSTM Training</title>
    <style>
        body { background: #111; color: #eee; font-family: system-ui, sans-serif; margin: 0; padding: 20px; }
        h1 { color: #7c8aff; margin: 0 0 10px 0; }
        .stats { color: #aaa; font-size: 14px; margin-bottom: 20px; }
        .stats span { margin-right: 20px; }
        .container { display: flex; gap: 20px; flex-wrap: wrap; }
        .plot-section { flex: 2; min-width: 400px; }
        .video-section { flex: 1; min-width: 300px; }
        .plot-section img { width: 100%; border-radius: 8px; background: #1a1a2e; }
        .video-list { display: flex; flex-direction: column; gap: 10px; }
        .video-item { background: #1a1a2e; border-radius: 8px; overflow: hidden; }
        .video-item video { width: 100%; display: block; }
        .video-item .label { padding: 8px; font-size: 12px; color: #888; }
        h2 { color: #888; font-size: 14px; margin: 0 0 10px 0; text-transform: uppercase; }
    </style>
</head>
<body>
    <h1>🧠 BeeWalker LSTM Training</h1>
    <div class="stats" id="stats">Loading...</div>
    <div class="container">
        <div class="plot-section">
            <h2>Training Progress</h2>
            <img id="plot" src="/plot" alt="Training plot">
        </div>
        <div class="video-section">
            <h2>Latest Videos</h2>
            <div class="video-list" id="videos">Loading...</div>
        </div>
    </div>
    <script>
        function update() {
            fetch('/api/stats').then(r=>r.json()).then(d=>{
                document.getElementById('stats').innerHTML = 
                    `<span>Steps: ${(d.steps/1e6).toFixed(3)}M</span>` +
                    `<span>Reward: ${d.reward.toFixed(1)}</span>` +
                    `<span>Best: ${d.best.toFixed(1)}</span>` +
                    `<span>Ep Len: ${d.ep_len.toFixed(0)}</span>` +
                    `<span style="color:#7c8aff">${d.fps.toFixed(0)} sps</span>`;
            });
            document.getElementById('plot').src = '/plot?' + Date.now();
            
            fetch('/api/videos').then(r=>r.json()).then(videos=>{
                const html = videos.map(v => 
                    `<div class="video-item">
                        <video controls muted loop playsinline>
                            <source src="/video/${v}" type="video/mp4">
                        </video>
                        <div class="label">${v}</div>
                    </div>`
                ).join('');
                document.getElementById('videos').innerHTML = html || '<div style="color:#666">No videos yet</div>';
            });
        }
        update();
        setInterval(update, 5000);  // Refresh every 5s
    </script>
</body>
</html>"""


@app.route('/plot')
def plot():
    plot_path = run_dir / 'progress.png'
    if plot_path.exists():
        return send_file(plot_path, mimetype='image/png')
    # Return placeholder
    return Response(b'', mimetype='image/png', status=204)


@app.route('/api/stats')
def api_stats():
    return json.dumps(stats)


@app.route('/api/videos')
def api_videos():
    video_dir = run_dir / 'videos'
    if video_dir.exists():
        videos = sorted([f.name for f in video_dir.glob('*.mp4')], reverse=True)[:5]
        return json.dumps(videos)
    return json.dumps([])


@app.route('/video/<filename>')
def video(filename):
    video_path = run_dir / 'videos' / filename
    if video_path.exists():
        return send_file(video_path, mimetype='video/mp4')
    return Response(b'', status=404)


# ============================================================================
# MAIN
# ============================================================================

def main():
    global run_dir
    
    parser = argparse.ArgumentParser(description="BeeWalker LSTM Training")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--hidden-size", type=int, default=32, help="LSTM hidden size (default: 32)")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel envs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--port", type=int, default=1306, help="Web UI port")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BEEWALKER_ROOT / f"results/lstm_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "videos").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    
    print("=" * 60)
    print("🧠 BeeWalker LSTM Training (RecurrentPPO)")
    print("=" * 60)
    print(f"  LSTM hidden size: {args.hidden_size}")
    print(f"  Parallel envs:    {args.n_envs}")
    print(f"  Learning rate:    {args.lr}")
    print(f"  Results:          {run_dir}")
    print(f"  Dashboard:        http://127.0.0.1:{args.port}")
    print("=" * 60)
    
    # Save run config
    run_config = {
        "lstm_hidden_size": args.hidden_size,
        "n_envs": args.n_envs,
        "learning_rate": args.lr,
        "reward_config": "speed",
        "timestamp": timestamp,
        "resume_from": args.resume,
    }
    with open(run_dir / "run_config.json", 'w') as f:
        json.dump(run_config, f, indent=2)
    
    # Start Flask
    flask_thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=args.port, threaded=True, use_reloader=False),
        daemon=True
    )
    flask_thread.start()
    
    # Create environments
    env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    eval_env = ConfigurableRewardEnv(config, render_mode="rgb_array")
    
    if args.resume:
        print(f"\n📂 Resuming from: {args.resume}")
        model = RecurrentPPO.load(args.resume, env=env, device="cpu")
        print(f"  Loaded successfully")
    else:
        print(f"\n🆕 Starting fresh LSTM training")
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=128,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.025,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="cpu",
            policy_kwargs=dict(
                lstm_hidden_size=args.hidden_size,
                n_lstm_layers=1,
                shared_lstm=False,
                enable_critic_lstm=True,
                net_arch=dict(pi=[64], vf=[64]),
            ),
            tensorboard_log=str(run_dir / "tensorboard"),
            verbose=1,
        )
    
    # Print model size
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"\n  📊 Model size: {total_params:,} params ({trainable_params:,} trainable)")
    print(f"  📊 Approx size: {total_params * 4 / 1024:.1f} KB (float32), {total_params / 1024:.1f} KB (int8)")
    
    # Callbacks
    train_cb = TrainingCallback(eval_env, run_dir / "videos", video_freq=100_000, plot_freq=10_000)
    checkpoint_cb = CheckpointCallback(
        save_freq=500_000 // args.n_envs,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="lstm"
    )
    
    print("\n🚀 Starting training... (Ctrl+C to stop)\n")
    try:
        model.learn(
            total_timesteps=1_000_000_000,
            callback=[train_cb, checkpoint_cb],
            progress_bar=True,
            reset_num_timesteps=(args.resume is None),
        )
    except KeyboardInterrupt:
        print("\n\n⏹ Training stopped by user.")
    
    # Cleanup
    model.save(str(run_dir / "final_model"))
    train_cb._save_reward_log()
    train_cb._update_plot()
    print(f"💾 Final model saved to: {run_dir / 'final_model'}")
    print(f"📈 Best reward: {train_cb.best_reward:.1f}")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
