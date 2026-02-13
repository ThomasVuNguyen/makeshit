"""
BeeWalker MJX Training Dashboard
Standalone Flask dashboard that monitors MJX training results.

Serves on port 1306 with:
  - Live reward/episode length charts
  - Training stats (steps, reward, FPS, ETA)
  - Periodic evaluation videos using CPU MuJoCo

Usage:
    python -m training.dashboard                           # Auto-detect latest mjx run
    python -m training.dashboard --run-dir results/mjx_*   # Specific run
"""

import os
import sys
import time
import json
import glob
import argparse
import threading
from pathlib import Path
from datetime import datetime

os.environ['MUJOCO_GL'] = 'egl'

_BEEWALKER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BEEWALKER_ROOT)

import numpy as np

try:
    from flask import Flask, Response, send_file
except ImportError:
    print("Installing flask...")
    os.system(f"{sys.executable} -m pip install --break-system-packages --quiet flask")
    from flask import Flask, Response, send_file

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
stats = {"steps": 0, "reward": 0, "best": 0, "ep_len": 0, "fps": 0, "eta_h": 0}
run_dir = None


# ============================================================================
# CHART GENERATION
# ============================================================================

def update_plot():
    """Read reward_log.json and generate progress.png."""
    if run_dir is None:
        return
    
    log_path = run_dir / "reward_log.json"
    if not log_path.exists():
        return
    
    try:
        with open(log_path) as f:
            data = json.load(f)
        
        if len(data) < 2:
            return
        
        steps = [d["steps"] for d in data]
        rewards = [d["mean_reward"] for d in data]
        lengths = [d["mean_length"] for d in data]
        fps_data = [d.get("fps", 0) for d in data]
        
        # Update stats
        latest = data[-1]
        stats["steps"] = latest["steps"]
        stats["reward"] = latest["mean_reward"]
        stats["ep_len"] = latest["mean_length"]
        stats["fps"] = latest.get("fps", 0)
        stats["best"] = max(rewards)
        
        # Read total_steps from config
        config_path = run_dir / "run_config.json"
        total_steps = 1_000_000_000
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
                total_steps = cfg.get("total_steps", total_steps)
        
        if stats["fps"] > 0:
            stats["eta_h"] = (total_steps - stats["steps"]) / stats["fps"] / 3600
        
        # ---- Generate plot ----
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), facecolor='#0d1117')
        
        for ax in axes:
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='#8b949e')
            ax.grid(True, alpha=0.15, color='#30363d')
            for s in ax.spines.values():
                s.set_color('#30363d')
        
        steps_k = np.array(steps) / 1e6
        
        # Reward plot
        axes[0].plot(steps_k, rewards, color='#58a6ff', linewidth=2, label='Mean Reward')
        # Smoothed line
        if len(rewards) > 10:
            window = min(20, len(rewards) // 3)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(steps_k[window-1:], smoothed, color='#3fb950', linewidth=2.5, 
                        alpha=0.9, label=f'Smoothed ({window})')
        
        best_val = max(rewards)
        axes[0].axhline(y=best_val, color='#f0883e', linestyle='--', alpha=0.6, linewidth=1)
        axes[0].text(steps_k[-1], best_val, f'  Best: {best_val:.1f}', color='#f0883e', 
                    fontsize=9, va='bottom')
        axes[0].set_ylabel('Mean Reward', color='#c9d1d9', fontsize=11)
        axes[0].legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=9)
        
        fps_text = f"{stats['fps']:,.0f} sps" if stats['fps'] > 0 else ""
        eta_text = f" · ETA: {stats['eta_h']:.1f}h" if stats['eta_h'] > 0 else ""
        axes[0].set_title(
            f'🚀 BeeWalker MJX Training — {stats["steps"]/1e6:.2f}M steps · {fps_text}{eta_text}',
            color='#c9d1d9', fontsize=13, fontweight='bold', pad=12
        )
        
        # Episode length plot
        axes[1].plot(steps_k, lengths, color='#bc8cff', linewidth=2)
        axes[1].set_ylabel('Episode Length', color='#c9d1d9', fontsize=11)
        axes[1].set_xlabel('Steps (M)', color='#c9d1d9', fontsize=11)
        
        plt.tight_layout(pad=2)
        plt.savefig(run_dir / 'progress.png', dpi=120, facecolor='#0d1117',
                    bbox_inches='tight', pad_inches=0.3)
        plt.close()
    except Exception as e:
        print(f"Plot error: {e}")


# ============================================================================
# VIDEO RECORDING (CPU MuJoCo)
# ============================================================================

def record_video(params_path, step_count):
    """Record evaluation video using CPU MuJoCo + saved params."""
    try:
        import mujoco
        import cv2
        import imageio
        import jax
        import jax.numpy as jnp
        
        from training.train_mjx import ActorCritic
        from env.bee_walker_env import BeeWalkerEnv
        
        print(f"  📹 Recording video at step {step_count:,}...")
        
        # Load and reconstruct params
        data = np.load(str(params_path), allow_pickle=True)
        params_tree = _unflatten_params(data)
        
        # Create CPU env for rendering
        env = BeeWalkerEnv(render_mode='rgb_array')
        
        # Create network
        network = ActorCritic(act_dim=6, hidden_size=32)
        
        obs, _ = env.reset()
        lstm_h = jnp.zeros(32)
        lstm_c = jnp.zeros(32)
        
        frames = []
        total_reward = 0.0
        n_frames = 300  # 10s at 30fps
        
        step_text = f"{step_count/1e6:.2f}M STEPS" if step_count >= 1e6 else f"{step_count/1e3:.0f}K STEPS"
        
        for i in range(n_frames):
            obs_jax = jnp.array(obs)
            mean, log_std, value, (lstm_h, lstm_c) = network.apply(
                {'params': params_tree}, obs_jax, (lstm_h, lstm_c))
            
            action = np.array(mean)  # Deterministic
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            
            frame = env.render()
            if frame is not None:
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 36), (12, 14, 20), -1)
                cv2.putText(frame, f"BEEWALKER MJX | {step_text} | R: {total_reward:.1f}", 
                           (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 190, 220), 1, cv2.LINE_AA)
                frames.append(frame)
            
            if term or trunc:
                obs, _ = env.reset()
                lstm_h = jnp.zeros(32)
                lstm_c = jnp.zeros(32)
                total_reward = 0.0
        
        env.close()
        
        if frames:
            video_dir = run_dir / "videos"
            video_dir.mkdir(exist_ok=True)
            video_path = video_dir / f"step_{step_count:09d}.mp4"
            imageio.mimwrite(str(video_path), frames, fps=30, codec='libx264',
                           output_params=['-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '18'])
            print(f"  📹 Saved: {video_path.name}")
    except Exception as e:
        import traceback
        print(f"  ⚠️  Video recording failed: {e}")
        traceback.print_exc()


def _unflatten_params(flat_data):
    """Convert flat numpy dict back to nested params dict.
    
    Keys are in JAX tree_leaves_with_path format: "['Dense_0']/['kernel']"
    We strip the brackets to get: Dense_0 / kernel
    """
    import re
    params = {}
    for key in flat_data.files:
        # Strip ['...'] brackets — keys look like "['Dense_0']/['kernel']"
        clean_key = re.sub(r"\['(.+?)'\]", r"\1", key)
        parts = clean_key.split('/')
        d = params
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = flat_data[key]
    return params


# ============================================================================
# BACKGROUND MONITOR
# ============================================================================

def monitor_loop(video_freq=500_000):
    """Background thread: updates plot + records videos periodically."""
    last_video_step = 0
    
    while True:
        time.sleep(10)
        update_plot()
        
        # Check if we should record a video
        if run_dir and stats["steps"] > 0:
            steps_since_video = stats["steps"] - last_video_step
            if steps_since_video >= video_freq:
                best_model = run_dir / "best_model.npz"
                if best_model.exists():
                    record_video(best_model, stats["steps"])
                    last_video_step = stats["steps"]


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    return """<!DOCTYPE html>
<html>
<head>
    <title>BeeWalker MJX Training</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0d1117; color: #c9d1d9; font-family: -apple-system, system-ui, sans-serif; padding: 24px; }
        h1 { color: #58a6ff; font-size: 22px; margin-bottom: 8px; }
        .subtitle { color: #8b949e; font-size: 13px; margin-bottom: 20px; }
        
        .stats-bar {
            display: flex; gap: 16px; flex-wrap: wrap;
            padding: 14px 20px; background: #161b22; border: 1px solid #30363d;
            border-radius: 8px; margin-bottom: 20px; align-items: center;
        }
        .stat { display: flex; flex-direction: column; }
        .stat .label { font-size: 10px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
        .stat .value { font-size: 18px; font-weight: 600; color: #c9d1d9; }
        .stat .value.highlight { color: #58a6ff; }
        .stat .value.green { color: #3fb950; }
        .stat .value.orange { color: #f0883e; }
        
        .container { display: flex; gap: 20px; flex-wrap: wrap; }
        .plot-section { flex: 2; min-width: 400px; }
        .video-section { flex: 1; min-width: 300px; }
        .plot-section img { width: 100%; border-radius: 8px; border: 1px solid #30363d; }
        .section-title { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
        
        .video-list { display: flex; flex-direction: column; gap: 10px; }
        .video-item { background: #161b22; border-radius: 8px; overflow: hidden; border: 1px solid #30363d; }
        .video-item video { width: 100%; display: block; }
        .video-item .label { padding: 8px 12px; font-size: 11px; color: #8b949e; }
        .no-data { color: #484f58; font-style: italic; font-size: 13px; padding: 20px; }
    </style>
</head>
<body>
    <h1>🚀 BeeWalker MJX Training</h1>
    <div class="subtitle">GPU-accelerated PPO+LSTM on RTX 4090</div>
    
    <div class="stats-bar" id="stats">
        <div class="stat"><span class="label">Loading...</span></div>
    </div>
    
    <div class="container">
        <div class="plot-section">
            <div class="section-title">Training Progress</div>
            <img id="plot" src="/plot" alt="Training plot">
        </div>
        <div class="video-section">
            <div class="section-title">Evaluation Videos</div>
            <div class="video-list" id="videos"><div class="no-data">Recording first video...</div></div>
        </div>
    </div>
    
    <script>
        function fmt(n) {
            if (n >= 1e6) return (n/1e6).toFixed(2) + 'M';
            if (n >= 1e3) return (n/1e3).toFixed(0) + 'K';
            return n.toFixed(0);
        }
        
        function update() {
            fetch('/api/stats').then(r=>r.json()).then(d=>{
                document.getElementById('stats').innerHTML = 
                    `<div class="stat"><span class="label">Steps</span><span class="value">${fmt(d.steps)}</span></div>` +
                    `<div class="stat"><span class="label">Reward</span><span class="value highlight">${d.reward.toFixed(1)}</span></div>` +
                    `<div class="stat"><span class="label">Best</span><span class="value green">${d.best.toFixed(1)}</span></div>` +
                    `<div class="stat"><span class="label">Ep Length</span><span class="value">${d.ep_len.toFixed(0)}</span></div>` +
                    `<div class="stat"><span class="label">Speed</span><span class="value highlight">${d.fps.toFixed(0)} sps</span></div>` +
                    `<div class="stat"><span class="label">ETA</span><span class="value orange">${d.eta_h.toFixed(1)}h</span></div>`;
            });
            document.getElementById('plot').src = '/plot?' + Date.now();
            
            fetch('/api/videos').then(r=>r.json()).then(videos=>{
                if (videos.length === 0) {
                    document.getElementById('videos').innerHTML = '<div class="no-data">Recording first video...</div>';
                    return;
                }
                const html = videos.map(v => 
                    `<div class="video-item">
                        <video controls muted loop playsinline>
                            <source src="/video/${v}" type="video/mp4">
                        </video>
                        <div class="label">${v.replace('.mp4','').replace('step_','Step ')}</div>
                    </div>`
                ).join('');
                document.getElementById('videos').innerHTML = html;
            });
        }
        update();
        setInterval(update, 5000);
    </script>
</body>
</html>"""


@app.route('/plot')
def plot():
    if run_dir:
        plot_path = run_dir / 'progress.png'
        if plot_path.exists():
            return send_file(plot_path, mimetype='image/png')
    return Response(b'', mimetype='image/png', status=204)


@app.route('/api/stats')
def api_stats():
    return json.dumps(stats)


@app.route('/api/videos')
def api_videos():
    if run_dir:
        video_dir = run_dir / 'videos'
        if video_dir.exists():
            videos = sorted([f.name for f in video_dir.glob('*.mp4')], reverse=True)[:5]
            return json.dumps(videos)
    return json.dumps([])


@app.route('/video/<filename>')
def video(filename):
    if run_dir:
        video_path = run_dir / 'videos' / filename
        if video_path.exists():
            return send_file(video_path, mimetype='video/mp4')
    return Response(b'', status=404)


# ============================================================================
# MAIN
# ============================================================================

def find_latest_run():
    """Find the most recent mjx results directory."""
    results_dir = Path(_BEEWALKER_ROOT) / "results"
    runs = sorted(results_dir.glob("mjx_*"), key=lambda p: p.name, reverse=True)
    if runs:
        return runs[0]
    return None


def main():
    global run_dir
    
    parser = argparse.ArgumentParser(description="BeeWalker MJX Training Dashboard")
    parser.add_argument("--run-dir", type=str, help="Results directory to monitor")
    parser.add_argument("--port", type=int, default=1306, help="Dashboard port (default: 1306)")
    parser.add_argument("--video-freq", type=int, default=500_000, help="Video recording frequency in steps")
    args = parser.parse_args()
    
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run()
    
    if run_dir is None:
        print("❌ No MJX training results found. Start training first.")
        sys.exit(1)
    
    print(f"{'='*50}")
    print(f"📊 BeeWalker MJX Dashboard")
    print(f"{'='*50}")
    print(f"  Monitoring: {run_dir}")
    print(f"  Dashboard:  http://0.0.0.0:{args.port}")
    print(f"  Video freq: every {args.video_freq:,} steps")
    print(f"{'='*50}")
    
    # Initial plot
    update_plot()
    
    # Start background monitor
    monitor = threading.Thread(target=monitor_loop, args=(args.video_freq,), daemon=True)
    monitor.start()
    
    # Start Flask
    app.run(host='0.0.0.0', port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
