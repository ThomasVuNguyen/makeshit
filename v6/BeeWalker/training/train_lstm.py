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
stats = {"steps": 0, "reward": 0, "best": 0, "ep_len": 0, "fps": 0, "curriculum": 0.0}
run_dir = None  # Set in main()


def make_env(rank, seed=0, env_kwargs=None):
    if env_kwargs is None:
        env_kwargs = {}

    def _init():
        from gymnasium.wrappers import TimeLimit
        env = ConfigurableRewardEnv(config, render_mode="rgb_array", **env_kwargs)
        env = TimeLimit(env, max_episode_steps=1000)  # Force episode end for stats
        env = Monitor(env)  # Track episode stats
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


class CurriculumCallback(BaseCallback):
    """Advances curriculum difficulty based on training progress.
    
    Single progress ramp from 0.0 (easy) to 1.0 (hard) over curriculum_steps.
    Controls ALL domain randomization ranges (friction, mass, motor, noise,
    pushes, gravity tilt) via the env's set_curriculum() method.
    """
    
    def __init__(self, curriculum_steps=50_000_000, update_freq=20_000):
        super().__init__()
        self.curriculum_steps = curriculum_steps
        self.update_freq = update_freq
        self._last_progress = -1.0
    
    def _on_step(self):
        if self.num_timesteps % self.update_freq == 0:
            progress = min(self.num_timesteps / self.curriculum_steps, 1.0)
            if abs(progress - self._last_progress) >= 0.01:
                self._last_progress = progress
                stats["curriculum"] = progress
                self.training_env.env_method("set_curriculum", progress)
                phase = min(int(progress * 4) + 1, 4)
                print(f"  📈 Curriculum: {progress:.0%} (Phase {phase}/4) — all DR ranges widening")
        return True


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
    
    @staticmethod
    def _add_hud(frame, step_text, reward_val, frame_idx, total_frames):
        """JARVIS-style HUD overlay."""
        h, w = frame.shape[:2]
        ov = frame.copy()
        # Top gradient
        for i in range(48):
            cv2.rectangle(ov, (0, i), (w, i+1), (12, 14, 20), -1)
        frame = cv2.addWeighted(ov, 0.5 * (1 - frame_idx * 0 ), frame, 0.5, 0)
        # Accent line
        cv2.line(frame, (20, 40), (w-20, 40), (45, 55, 70), 1)
        cx = w // 2
        cv2.line(frame, (cx-80, 40), (cx+80, 40), (70, 140, 200), 2)
        # Title
        cv2.putText(frame, "BEEWALKER", (24, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 150, 180), 1, cv2.LINE_AA)
        # Step counter
        ts = cv2.getTextSize(step_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.putText(frame, step_text, (cx - ts[0]//2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 190, 220), 1, cv2.LINE_AA)
        # Reward
        rt = f"R {reward_val:.1f}"
        rc = (80, 210, 140) if reward_val > 0 else (210, 140, 80)
        ts2 = cv2.getTextSize(rt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(frame, rt, (w - ts2[0] - 24, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rc, 1, cv2.LINE_AA)
        # Bottom progress bar
        ov2 = frame.copy()
        for i in range(30):
            cv2.rectangle(ov2, (0, h-30+i), (w, h-30+i+1), (12, 14, 20), -1)
        frame = cv2.addWeighted(ov2, 0.35, frame, 0.65, 0)
        by = h - 12
        p = frame_idx / max(total_frames, 1)
        cv2.line(frame, (24, by), (w-24, by), (35, 42, 55), 2)
        be = 24 + int((w - 48) * p)
        cv2.line(frame, (24, by), (be, by), (60, 130, 210), 2)
        cv2.circle(frame, (be, by), 3, (100, 190, 255), -1)
        # Time
        cv2.putText(frame, f"{frame_idx/30:.1f}s", (24, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 95, 120), 1, cv2.LINE_AA)
        # Corner brackets
        bc = (40, 55, 72)
        for (x1,y1),(x2,y2),(x3,y3) in [((8,8),(28,8),(8,28)),((w-8,8),(w-28,8),(w-8,28)),((8,h-8),(28,h-8),(8,h-28)),((w-8,h-8),(w-28,h-8),(w-8,h-28))]:
            cv2.line(frame,(x1,y1),(x2,y2),bc,1); cv2.line(frame,(x1,y1),(x3,y3),bc,1)
        return frame

    def _record_video(self):
        """Record evaluation video using deterministic policy with JARVIS HUD."""
        obs = self.eval_env.reset()[0]
        frames = []
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        total_reward = 0.0
        n_frames = 300  # 10 seconds at 30fps
        
        # Format step count for HUD
        steps = self.num_timesteps
        if steps >= 1_000_000:
            step_text = f"{steps/1_000_000:.2f}M STEPS"
        elif steps >= 1_000:
            step_text = f"{steps/1_000:.0f}K STEPS"
        else:
            step_text = f"{steps:,} STEPS"
        
        for i in range(n_frames):
            action, lstm_states = self.model.predict(
                obs, state=lstm_states, 
                episode_start=episode_starts,
                deterministic=True
            )
            obs, reward, term, trunc, _ = self.eval_env.step(action)
            total_reward += reward
            frame = self.eval_env.render()
            if frame is not None:
                frame = self._add_hud(frame, step_text, total_reward, i, n_frames)
                frames.append(frame)
            
            episode_starts = np.array([term or trunc])
            if term or trunc:
                obs = self.eval_env.reset()[0]
                total_reward = 0.0
        
        if frames:
            video_dir = self.save_dir
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"step_{self.num_timesteps:09d}.mp4"
            imageio.mimwrite(
                str(video_path), frames, fps=30, codec='libx264',
                output_params=['-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '18']
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

def _parse_arch(arch_str: str, arg_name: str):
    parts = [p.strip() for p in arch_str.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{arg_name} cannot be empty")
    arch = []
    for p in parts:
        try:
            v = int(p)
        except ValueError as exc:
            raise ValueError(f"{arg_name} must be comma-separated ints, got '{arch_str}'") from exc
        if v <= 0:
            raise ValueError(f"{arg_name} values must be > 0, got {v}")
        arch.append(v)
    return arch


def main():
    global run_dir
    
    parser = argparse.ArgumentParser(description="BeeWalker LSTM Training")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--hidden-size", type=int, default=32, help="LSTM hidden size (default: 32)")
    parser.add_argument("--n-envs", type=int, default=10, help="Number of parallel envs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--port", type=int, default=1306, help="Web UI port")
    parser.add_argument("--curriculum-steps", type=int, default=50_000_000,
                        help="Steps over which to ramp all domain randomization (default: 50M)")
    parser.add_argument("--residual-action", action="store_true",
                        help="Use residual actions around a phase-based gait reference")
    parser.add_argument("--phase-features", action="store_true",
                        help="Append sin/cos phase features to observations")
    parser.add_argument("--residual-limit", type=float, default=0.35,
                        help="Residual action limit in radians when residual mode is enabled")
    parser.add_argument("--reference-gait-scale", type=float, default=1.0,
                        help="Scale factor for the built-in phase reference gait")
    parser.add_argument("--gait-frequency-hz", type=float, default=2.0,
                        help="Reference gait frequency in Hz")
    parser.add_argument("--teacher-size", choices=["none", "medium", "large"], default="none",
                        help="Teacher preset: medium=(LSTM64, heads 128,128), large=(LSTM128, heads 256,128)")
    parser.add_argument("--pi-arch", type=str, default="64",
                        help="Policy MLP head sizes, comma-separated (e.g. 256,128)")
    parser.add_argument("--vf-arch", type=str, default="64",
                        help="Value MLP head sizes, comma-separated (e.g. 256,128)")
    args = parser.parse_args()

    # Optional teacher presets for larger-capacity policies.
    if args.teacher_size == "medium":
        args.hidden_size = 64
        args.pi_arch = "128,128"
        args.vf_arch = "128,128"
    elif args.teacher_size == "large":
        args.hidden_size = 128
        args.pi_arch = "256,128"
        args.vf_arch = "256,128"

    pi_arch = _parse_arch(args.pi_arch, "--pi-arch")
    vf_arch = _parse_arch(args.vf_arch, "--vf-arch")

    env_kwargs = {
        "residual_action": args.residual_action,
        "phase_features": args.phase_features,
        "residual_limit": args.residual_limit,
        "reference_gait_scale": args.reference_gait_scale,
        "gait_frequency_hz": args.gait_frequency_hz,
    }
    
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
    print(f"  Curriculum ramp:  {args.curriculum_steps/1e6:.0f}M steps (all DR)")
    print(f"  Residual action:  {args.residual_action}")
    print(f"  Phase features:   {args.phase_features}")
    if args.residual_action:
        print(f"  Residual limit:   {args.residual_limit:.3f} rad")
    print(f"  Gait ref scale:   {args.reference_gait_scale:.3f}")
    print(f"  Gait ref freq:    {args.gait_frequency_hz:.2f} Hz")
    print(f"  Teacher preset:   {args.teacher_size}")
    print(f"  Pi arch:          {pi_arch}")
    print(f"  Vf arch:          {vf_arch}")
    print("=" * 60)
    
    # Save run config
    run_config = {
        "lstm_hidden_size": args.hidden_size,
        "pi_arch": pi_arch,
        "vf_arch": vf_arch,
        "teacher_size": args.teacher_size,
        "n_envs": args.n_envs,
        "learning_rate": args.lr,
        "reward_config": "speed",
        "timestamp": timestamp,
        "resume_from": args.resume,
        "residual_action": args.residual_action,
        "phase_features": args.phase_features,
        "residual_limit": args.residual_limit,
        "reference_gait_scale": args.reference_gait_scale,
        "gait_frequency_hz": args.gait_frequency_hz,
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
    env = SubprocVecEnv([make_env(i, env_kwargs=env_kwargs) for i in range(args.n_envs)])
    eval_env = ConfigurableRewardEnv(config, render_mode="rgb_array", **env_kwargs)
    
    if args.resume:
        print(f"\n📂 Resuming from: {args.resume}")
        model = RecurrentPPO.load(args.resume, env=env, device="cpu")
        # FIX: SB3 .load() ignores the --lr arg and uses the saved model's LR.
        # We must explicitly override it after loading.
        model.learning_rate = args.lr
        model.lr_schedule = lambda _: args.lr  # Constant LR schedule
        print(f"  Loaded successfully (LR overridden to {args.lr})")
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
                net_arch=dict(pi=pi_arch, vf=vf_arch),
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
    curriculum_cb = CurriculumCallback(curriculum_steps=args.curriculum_steps)
    checkpoint_cb = CheckpointCallback(
        save_freq=500_000 // args.n_envs,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="lstm"
    )
    
    print("\n🚀 Starting training... (Ctrl+C to stop)\n")
    try:
        model.learn(
            total_timesteps=1_000_000_000,
            callback=[train_cb, curriculum_cb, checkpoint_cb],
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
