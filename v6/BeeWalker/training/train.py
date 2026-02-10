"""
BeeWalker 24-Hour Training Script
Runs multiple parallel experiments with different reward configurations.
Designed to maximize learning over 24 hours of compute.

Usage:
    python train.py                    # Run full 24-hour experiment suite
    python train.py --phase 1          # Run only Phase 1 (reward sweep)
    python train.py --phase 2          # Run Phase 2 (long training on best)
    python train.py --quick            # Quick test (1M steps per config)
"""
import os
import sys
os.environ['MUJOCO_GL'] = 'egl'

# Add BeeWalker root to path for cross-package imports
_BEEWALKER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BEEWALKER_ROOT)

import argparse
import time
import threading
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import shutil

import numpy as np
import cv2
from flask import Flask, Response, render_template_string
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from env.bee_walker_env import BeeWalkerEnv


# ============================================================================
# REWARD CONFIGURATIONS
# ============================================================================

@dataclass
class RewardConfig:
    """Configuration for different reward strategies."""
    name: str
    description: str
    
    # Reward weights
    velocity_weight: float = 2.0
    upright_weight: float = 0.5
    height_weight: float = 0.5
    stepping_weight: float = 2.0
    ctrl_cost_weight: float = 0.001
    drift_penalty_weight: float = 0.1
    
    # Additional reward components
    knee_bend_weight: float = 0.0      # Reward knee movement
    foot_clearance_weight: float = 0.0  # Reward lifting feet
    symmetry_weight: float = 0.0        # Reward gait symmetry
    efficiency_weight: float = 0.0      # Reward energy efficiency
    exploration_bonus: float = 0.0      # Entropy bonus
    
    # Training parameters
    total_timesteps: int = 10_000_000
    ent_coef: float = 0.01
    learning_rate: float = 3e-4


# Six different reward profiles for Phase 1
REWARD_CONFIGS = {
    "natural": RewardConfig(
        name="natural",
        description="Forces knee bending and foot clearance for natural gait",
        knee_bend_weight=3.0,
        foot_clearance_weight=5.0,
        stepping_weight=3.0,
        ctrl_cost_weight=0.0005,
    ),
    "speed": RewardConfig(
        name="speed",
        description="Maximum forward velocity focus",
        velocity_weight=5.0,
        upright_weight=0.3,
        stepping_weight=1.0,
        ctrl_cost_weight=0.0001,
    ),
    "explorer": RewardConfig(
        name="explorer",
        description="High exploration with curiosity-like bonus",
        exploration_bonus=0.5,
        ent_coef=0.05,  # Much higher entropy
        velocity_weight=1.5,
        stepping_weight=3.0,
    ),
    "efficient": RewardConfig(
        name="efficient",
        description="Energy-efficient locomotion",
        efficiency_weight=2.0,
        velocity_weight=1.0,
        ctrl_cost_weight=0.01,  # Strong penalty for effort
    ),
    "symmetric": RewardConfig(
        name="symmetric",
        description="Enforces left-right symmetry for natural walking",
        symmetry_weight=3.0,
        knee_bend_weight=2.0,
        stepping_weight=2.0,
    ),
    "aggressive": RewardConfig(
        name="aggressive",
        description="Risk-taking with high velocity and low safety margin",
        velocity_weight=8.0,
        upright_weight=0.2,
        height_weight=0.1,
        stepping_weight=4.0,
        ent_coef=0.02,
    ),
}


# ============================================================================
# CUSTOM ENVIRONMENT WITH CONFIGURABLE REWARDS
# ============================================================================

class ConfigurableRewardEnv(BeeWalkerEnv):
    """BeeWalker environment with configurable reward function."""
    
    def __init__(self, reward_config: RewardConfig, render_mode=None, max_episode_steps=1000):
        super().__init__(render_mode=render_mode, max_episode_steps=max_episode_steps)
        self.reward_config = reward_config
        self._prev_joint_pos = None
        self._prev_left_foot_z = 0
        self._prev_right_foot_z = 0
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_joint_pos = self.data.qpos[7:13].copy()  # After freejoint
        self._prev_left_foot_z = self.data.body("left_foot").xpos[2]
        self._prev_right_foot_z = self.data.body("right_foot").xpos[2]
        return obs, info
    
    def _compute_reward(self, action):
        """Enhanced reward function with configurable components."""
        cfg = self.reward_config
        
        pelvis_pos = self.data.body("pelvis").xpos
        pelvis_vel = self.data.body("pelvis").cvel
        pelvis_mat = self.data.body("pelvis").xmat.reshape(3, 3)
        upright = pelvis_mat[2, 2]
        
        # Base rewards
        forward_vel = pelvis_vel[3]
        velocity_reward = forward_vel * cfg.velocity_weight
        
        # Standing-still penalty — make standing unprofitable
        if abs(forward_vel) < 0.05:
            velocity_reward = -2.0
        
        upright_reward = upright * cfg.upright_weight
        height_reward = (cfg.height_weight if pelvis_pos[2] > 0.15 else 0.0)
        
        # Foot positions
        left_foot_z = self.data.body("left_foot").xpos[2]
        right_foot_z = self.data.body("right_foot").xpos[2]
        foot_diff = abs(left_foot_z - right_foot_z)
        stepping_reward = foot_diff * cfg.stepping_weight
        
        # Control cost
        ctrl_cost = cfg.ctrl_cost_weight * np.sum(action**2)
        
        # Drift penalty
        lateral_vel = abs(pelvis_vel[4])
        drift_penalty = cfg.drift_penalty_weight * lateral_vel
        
        # === NEW REWARD COMPONENTS ===
        
        # Knee bend reward - penalize straight knees
        knee_bend_reward = 0.0
        if cfg.knee_bend_weight > 0:
            left_knee = abs(self.data.qpos[self._joint_qpos_indices[1]])
            right_knee = abs(self.data.qpos[self._joint_qpos_indices[4]])
            knee_bend_reward = cfg.knee_bend_weight * (left_knee + right_knee) / 2
        
        # Foot clearance reward - reward lifting feet high
        foot_clearance_reward = 0.0
        if cfg.foot_clearance_weight > 0:
            max_foot_height = max(left_foot_z, right_foot_z)
            if max_foot_height > 0.02:  # Must lift at least 2cm
                foot_clearance_reward = cfg.foot_clearance_weight * (max_foot_height - 0.02)
        
        # Gait symmetry reward
        symmetry_reward = 0.0
        if cfg.symmetry_weight > 0:
            # Check if legs are in opposite phases
            left_hip = self.data.qpos[self._joint_qpos_indices[0]]
            right_hip = self.data.qpos[self._joint_qpos_indices[3]]
            # Opposite signs = good symmetry
            symmetry = -left_hip * right_hip  # Positive when opposite
            symmetry_reward = cfg.symmetry_weight * max(0, symmetry)
        
        # Energy efficiency
        efficiency_reward = 0.0
        if cfg.efficiency_weight > 0:
            # Reward distance per unit energy
            if np.sum(action**2) > 0.01:
                efficiency = forward_vel / (np.sum(action**2) + 0.1)
                efficiency_reward = cfg.efficiency_weight * efficiency
        
        # Exploration bonus (state-based novelty approximation)
        exploration_bonus = 0.0
        if cfg.exploration_bonus > 0:
            # Reward joint velocity (movement = exploration)
            joint_vel = np.sum(np.abs(self.data.qvel[6:12]))
            exploration_bonus = cfg.exploration_bonus * joint_vel
        
        # === REFERENCE MOTION REWARD ===
        # Sine-wave walking reference at 2Hz — soft bonus for tracking
        phase = (self._step_count / 50.0) * 2.0 * np.pi * 2.0  # 2Hz gait at 50Hz control
        ref_joints = np.array([
            0.4 * np.sin(phase),             # left hip
           -0.3 * np.cos(phase),             # left knee
            0.1 * np.sin(phase),             # left ankle
           -0.4 * np.sin(phase),             # right hip (anti-phase)
           -0.3 * np.cos(phase + np.pi),     # right knee (anti-phase)
           -0.1 * np.sin(phase),             # right ankle (anti-phase)
        ])
        joint_pos = np.array([self.data.qpos[i] for i in self._joint_qpos_indices])
        ref_error = np.sum((joint_pos - ref_joints) ** 2)
        reference_reward = 1.0 * np.exp(-2.0 * ref_error)  # 0 to 1.0 bonus
        
        # Survival bonus — small reward per timestep alive
        survival_bonus = 0.1
        
        total_reward = (
            velocity_reward +
            upright_reward +
            height_reward +
            stepping_reward +
            knee_bend_reward +
            foot_clearance_reward +
            symmetry_reward +
            efficiency_reward +
            exploration_bonus +
            reference_reward +
            survival_bonus -
            ctrl_cost -
            drift_penalty
        )
        
        return total_reward


# ============================================================================
# TRAINING INFRASTRUCTURE
# ============================================================================

# Flask app for multi-experiment visualization
app = Flask(__name__)
experiment_frames = {}
experiment_stats = {}
frame_lock = threading.Lock()


def make_configurable_env(reward_config: RewardConfig, rank: int, seed: int = 0):
    """Factory for creating configurable environments."""
    def _init():
        env = ConfigurableRewardEnv(reward_config, render_mode="rgb_array")
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


class ExperimentCallback(BaseCallback):
    """Callback for tracking experiment progress."""
    
    def __init__(self, config_name: str, eval_env, video_freq: int = 100000, verbose: int = 1):
        super().__init__(verbose)
        self.config_name = config_name
        self.eval_env = eval_env
        self.video_freq = video_freq
        self.best_mean_reward = float('-inf')
        self.rewards_history = []
    
    def _on_step(self):
        global experiment_frames, experiment_stats
        
        # Update stats
        with frame_lock:
            experiment_stats[self.config_name] = {
                "timesteps": self.num_timesteps,
                "best_reward": self.best_mean_reward,
            }
        
        # Render frame for visualization
        if self.num_timesteps % 500 == 0:
            try:
                frame = self.training_env.env_method("render")[0]
                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    text = f"{self.config_name}: {self.num_timesteps//1000}k steps"
                    cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    ret, buffer = cv2.imencode('.jpg', frame_bgr)
                    if ret:
                        with frame_lock:
                            experiment_frames[self.config_name] = buffer.tobytes()
            except:
                pass
        
        # Record video
        if self.num_timesteps % self.video_freq == 0 and self.num_timesteps > 0:
            self._record_video()
        
        return True
    
    def _on_rollout_end(self):
        if len(self.model.ep_info_buffer) > 0:
            rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            mean_reward = np.mean(rewards)
            self.rewards_history.append((self.num_timesteps, mean_reward))
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
    
    def _record_video(self):
        """Record evaluation video."""
        run_dir = self.model.logger.dir
        if run_dir is None:
            return
        
        video_dir = Path(run_dir) / "videos"
        video_dir.mkdir(exist_ok=True)
        
        obs, _ = self.eval_env.reset()
        frames = []
        
        for _ in range(300):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            frame = self.eval_env.render()
            if frame is not None:
                frames.append(frame)
            if terminated or truncated:
                obs, _ = self.eval_env.reset()
        
        if frames:
            video_path = video_dir / f"step_{self.num_timesteps:08d}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 30, (640, 480))
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()


def run_experiment(config_name: str, config: RewardConfig, base_dir: Path, n_envs: int = 4):
    """Run a single experiment configuration."""
    print(f"\n{'='*60}")
    print(f"Starting experiment: {config_name}")
    print(f"Description: {config.description}")
    print(f"Timesteps: {config.total_timesteps:,}")
    print(f"{'='*60}\n")
    
    # Create experiment directory
    exp_dir = base_dir / config_name
    exp_dir.mkdir(exist_ok=True)
    (exp_dir / "videos").mkdir(exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    
    # Save config
    with open(exp_dir / "config.json", 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Create environments
    env = SubprocVecEnv([make_configurable_env(config, i) for i in range(n_envs)])
    eval_env = ConfigurableRewardEnv(config, render_mode="rgb_array")
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=config.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        tensorboard_log=str(exp_dir / "tensorboard"),
        verbose=1,
    )
    
    # Callbacks
    exp_callback = ExperimentCallback(config_name, eval_env, video_freq=config.total_timesteps // 20)
    checkpoint_callback = CheckpointCallback(
        save_freq=config.total_timesteps // (10 * n_envs),
        save_path=str(exp_dir / "checkpoints"),
        name_prefix=config_name
    )
    
    # Train
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=[exp_callback, checkpoint_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print(f"\n{config_name}: Training interrupted")
    
    elapsed = time.time() - start_time
    
    # Save final model and results
    model.save(str(exp_dir / "final_model"))
    
    results = {
        "config_name": config_name,
        "total_timesteps": config.total_timesteps,
        "elapsed_seconds": elapsed,
        "best_mean_reward": exp_callback.best_mean_reward,
        "rewards_history": exp_callback.rewards_history,
    }
    with open(exp_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{config_name}: Completed in {elapsed/60:.1f} min, best reward: {exp_callback.best_mean_reward:.2f}")
    
    env.close()
    eval_env.close()
    
    return results


def run_parallel_experiments(configs: Dict[str, RewardConfig], base_dir: Path, max_parallel: int = 3):
    """Run multiple experiments, up to max_parallel at a time."""
    import concurrent.futures
    
    results = {}
    config_items = list(configs.items())
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for name, config in config_items:
            future = executor.submit(run_experiment, name, config, base_dir)
            futures[future] = name
        
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                print(f"Experiment {name} failed: {e}")
                results[name] = {"error": str(e)}
    
    return results


# ============================================================================
# WEB VISUALIZATION
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>BeeWalker Experiments</title>
    <style>
        body { font-family: Arial; background: #1a1a1a; color: white; margin: 20px; }
        .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
        .experiment { background: #2a2a2a; padding: 15px; border-radius: 10px; }
        .experiment h3 { margin: 0 0 10px 0; color: #4CAF50; }
        img { width: 100%; border-radius: 5px; }
        .stats { font-size: 14px; color: #aaa; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>🐝 BeeWalker 24-Hour Training</h1>
    <div class="grid">
        {% for name in experiments %}
        <div class="experiment">
            <h3>{{ name }}</h3>
            <img src="/feed/{{ name }}" />
            <div class="stats" id="stats-{{ name }}">Loading...</div>
        </div>
        {% endfor %}
    </div>
    <script>
        setInterval(() => {
            fetch('/stats').then(r => r.json()).then(data => {
                for (let name in data) {
                    let el = document.getElementById('stats-' + name);
                    if (el) el.innerText = `Steps: ${(data[name].timesteps/1000).toFixed(0)}k | Best: ${data[name].best_reward.toFixed(1)}`;
                }
            });
        }, 2000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, experiments=list(REWARD_CONFIGS.keys()))

@app.route('/feed/<name>')
def feed(name):
    def generate():
        while True:
            with frame_lock:
                frame = experiment_frames.get(name)
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    with frame_lock:
        return json.dumps(experiment_stats)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="BeeWalker 24-Hour Training")
    parser.add_argument("--phase", type=int, default=0, help="Run specific phase (1-4), 0=all")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (1M steps)")
    parser.add_argument("--config", type=str, help="Run single config by name")
    parser.add_argument("--no-viz", action="store_true", help="Disable web visualization")
    parser.add_argument("--parallel", type=int, default=2, help="Max parallel experiments")
    args = parser.parse_args()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(_BEEWALKER_ROOT) / "results" / f"sweep_{timestamp}"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🐝 BeeWalker 24-Hour Training Suite")
    print("="*60)
    print(f"Results directory: {base_dir}")
    print(f"Parallel experiments: {args.parallel}")
    
    # Adjust timesteps for quick mode
    configs = REWARD_CONFIGS.copy()
    if args.quick:
        print("Quick mode: 1M steps per config")
        for cfg in configs.values():
            cfg.total_timesteps = 1_000_000
    
    # Single config mode
    if args.config:
        if args.config not in configs:
            print(f"Unknown config: {args.config}")
            print(f"Available: {list(configs.keys())}")
            return
        configs = {args.config: configs[args.config]}
    
    # Start web server
    if not args.no_viz:
        flask_thread = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=1306, threaded=True, use_reloader=False),
            daemon=True
        )
        flask_thread.start()
        print(f"\n📺 Web dashboard: http://127.0.0.1:1306\n")
    
    # Run experiments
    print(f"\nRunning {len(configs)} experiments: {list(configs.keys())}\n")
    
    if args.parallel == 1:
        # Sequential execution
        all_results = {}
        for name, config in configs.items():
            all_results[name] = run_experiment(name, config, base_dir)
    else:
        # Parallel execution
        all_results = run_parallel_experiments(configs, base_dir, max_parallel=args.parallel)
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for name, results in all_results.items():
        if "error" in results:
            print(f"  {name}: FAILED - {results['error']}")
        else:
            print(f"  {name}: Best reward = {results['best_mean_reward']:.2f}")
    
    # Find best
    best_name = max(all_results.keys(), key=lambda k: all_results[k].get('best_mean_reward', float('-inf')))
    print(f"\n🏆 Best experiment: {best_name}")
    print(f"   Results saved to: {base_dir}")


if __name__ == "__main__":
    main()
