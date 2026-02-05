"""
Continuous training for speed config with live streaming.
Runs until Ctrl+C, saves checkpoints every 1M steps.
Watch at http://127.0.0.1:1306

FIX: Added log_std clamping to prevent NaN from exploding std.
"""
import os
os.environ['MUJOCO_GL'] = 'egl'

import threading
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, Response
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
import numpy as np
import cv2
import imageio

from bee_walker_env import BeeWalkerEnv
from train import ConfigurableRewardEnv, REWARD_CONFIGS


class StableActorCriticPolicy(ActorCriticPolicy):
    """
    Custom policy that clamps log_std to prevent numerical instability.
    log_std is clamped to [-20, 2] which corresponds to std in [~2e-9, ~7.4].
    This prevents the exploding std issue that caused NaN values.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Clamp initial log_std to a tighter range for stability
        with torch.no_grad():
            self.log_std.data.clamp_(-2.0, 0.0)
    
    def forward(self, obs, deterministic=False):
        """Override forward to clamp log_std before action distribution."""
        # Tighter clamp: std range [0.14, 2.7] instead of [~0, 7.4]
        with torch.no_grad():
            self.log_std.data.clamp_(-2.0, 1.0)
        return super().forward(obs, deterministic)
    
    def evaluate_actions(self, obs, actions):
        """Override to clamp log_std during evaluation."""
        with torch.no_grad():
            self.log_std.data.clamp_(-2.0, 1.0)
        return super().evaluate_actions(obs, actions)
    
    def _get_action_dist_from_latent(self, latent_pi):
        """Override to clamp log_std when getting action distribution."""
        with torch.no_grad():
            self.log_std.data.clamp_(-2.0, 1.0)
        return super()._get_action_dist_from_latent(latent_pi)

# Use speed config
config = REWARD_CONFIGS["speed"]

# Flask app for streaming
app = Flask(__name__)
latest_frame = None
frame_lock = threading.Lock()
stats = {"steps": 0, "reward": 0, "best": 0}

def make_env(rank, seed=0):
    def _init():
        env = ConfigurableRewardEnv(config, render_mode="rgb_array")
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

class StreamingCallback(BaseCallback):
    def __init__(self, eval_env, save_dir, video_freq=1_000_000):
        super().__init__()
        self.eval_env = eval_env
        self.save_dir = Path(save_dir)
        self.video_freq = video_freq
        self.best_reward = float('-inf')
    
    def _on_step(self):
        global latest_frame, stats
        
        stats["steps"] = self.num_timesteps
        
        # Stream frame every 100 steps
        if self.num_timesteps % 100 == 0:
            try:
                frame = self.training_env.env_method("render")[0]
                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.putText(frame_bgr, f"Steps: {self.num_timesteps:,}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame_bgr, f"Reward: {stats['reward']:.1f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame_bgr, f"Best: {stats['best']:.1f}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    ret, buffer = cv2.imencode('.jpg', frame_bgr)
                    if ret:
                        with frame_lock:
                            latest_frame = buffer.tobytes()
            except:
                pass
        
        # Save video every 1M steps
        if self.num_timesteps % self.video_freq == 0 and self.num_timesteps > 0:
            self._record_video()
        
        return True
    
    def _on_rollout_end(self):
        global stats
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            stats["reward"] = mean_reward
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                stats["best"] = mean_reward
                print(f"  New best: {mean_reward:.1f}")
    
    def _record_video(self):
        obs, _ = self.eval_env.reset()
        frames = []
        for _ in range(150):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = self.eval_env.step(action)
            frame = self.eval_env.render()
            if frame is not None:
                frames.append(frame)
            if term or trunc:
                obs, _ = self.eval_env.reset()
        
        if frames:
            video_path = self.save_dir / f"step_{self.num_timesteps:08d}.mp4"
            imageio.mimwrite(str(video_path), frames, fps=30, codec='libx264',
                           output_params=['-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '23'])
            print(f"  Saved: {video_path}")

@app.route('/')
def index():
    return """<html><body style="background:#111;margin:0">
    <img src="/stream" style="width:100%;height:100vh;object-fit:contain"/>
    </body></html>"""

@app.route('/stream')
def stream():
    def gen():
        while True:
            with frame_lock:
                frame = latest_frame
            if frame:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            time.sleep(0.033)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/speed_continuous_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "videos").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    
    print("="*60)
    print("SPEED CONFIG - CONTINUOUS TRAINING")
    print(f"Results: {run_dir}")
    print("Watch at: http://127.0.0.1:1306")
    print("Press Ctrl+C to stop")
    print("="*60)
    
    # Start Flask
    flask_thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=1306, threaded=True, use_reloader=False),
        daemon=True
    )
    flask_thread.start()
    
    n_envs = 4
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    eval_env = ConfigurableRewardEnv(config, render_mode="rgb_array")
    
    prev_checkpoint = "results/speed_continuous_20260203_105649/checkpoints/speed_99000000_steps"
    print(f"\nLoading from: {prev_checkpoint}")
    model = PPO.load(
        prev_checkpoint, 
        env=env, 
        device="cpu",
        custom_objects={"policy_class": StableActorCriticPolicy}
    )
    
    # CRITICAL: Reset the corrupted log_std from the checkpoint
    # The old checkpoint had log_std values that exploded to ~6e17
    with torch.no_grad():
        print(f"Old log_std: {model.policy.log_std.data}")
        model.policy.log_std.data.clamp_(-2.0, 0.5)  # Reset to reasonable range
        print(f"Reset log_std to: {model.policy.log_std.data}")
    
    stream_cb = StreamingCallback(eval_env, run_dir / "videos", video_freq=1_000_000)
    checkpoint_cb = CheckpointCallback(
        save_freq=1_000_000 // n_envs,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="speed"
    )
    
    print("\nStarting training...\n")
    try:
        model.learn(
            total_timesteps=1_000_000_000,
            callback=[stream_cb, checkpoint_cb],
            progress_bar=True,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\n\nTraining stopped by user.")
    
    model.save(str(run_dir / "final_model"))
    print(f"Final model saved to: {run_dir / 'final_model'}")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
