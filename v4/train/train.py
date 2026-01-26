"""
BeeWalker Training Script
Run with: python train.py

Trains a bipedal walking robot using PPO with live visualization on port 1306.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def main():
    print("=" * 60)
    print("🐝 BeeWalker Training System")
    print("=" * 60)
    print()
    
    # Import dependencies
    print("📦 Loading dependencies...")
    
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    from env import BeeWalkerEnv
    from app import start_server_thread, emit_metrics, emit_frame, emit_reference_frame, training_state
    
    print("✅ Dependencies loaded")
    print()
    
    # Start web server
    print("🌐 Starting training UI server...")
    start_server_thread(port=1306)
    time.sleep(2)  # Give server time to start
    
    # Create environment
    print("🤖 Creating BeeWalker environment...")
    
    def make_env():
        return BeeWalkerEnv(render_mode="rgb_array")
    
    env = DummyVecEnv([make_env])
    print("✅ Environment created")
    print()
    
    # Create models directory
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Training callback for visualization
    class TrainingCallback(BaseCallback):
        def __init__(self, render_freq=10, verbose=0):
            super().__init__(verbose)
            self.render_freq = render_freq
            self.episode_rewards = []
            self.current_episode_reward = 0
            self.episode_count = 0
            self.best_mean_reward = float('-inf')
            
        def _on_step(self) -> bool:
            # Check if training should stop
            if not training_state['is_running']:
                return False
            
            # Handle pause
            while training_state['is_paused'] and training_state['is_running']:
                time.sleep(0.1)
            
            # Track rewards
            if len(self.locals.get('rewards', [])) > 0:
                self.current_episode_reward += self.locals['rewards'][0]
            
            # Check for episode end
            if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
                self.episode_count += 1
                self.episode_rewards.append(self.current_episode_reward)
                
                # Calculate mean of last 10 episodes
                recent_rewards = self.episode_rewards[-10:]
                mean_reward = np.mean(recent_rewards)
                
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Save best model
                    self.model.save(models_dir / "best_model")
                
                # Emit metrics
                emit_metrics({
                    'is_running': True,
                    'is_paused': training_state['is_paused'],
                    'total_timesteps': self.num_timesteps,
                    'episodes': self.episode_count,
                    'episode_reward': self.current_episode_reward,
                    'mean_reward': mean_reward,
                    'best_reward': self.best_mean_reward,
                })
                
                self.current_episode_reward = 0
            
            # Render frames periodically
            if self.n_calls % self.render_freq == 0:
                try:
                    # Main tracking view
                    frame = env.envs[0].render()
                    if frame is not None:
                        emit_frame(frame)
                    
                    # Reference view (fixed angle for model preview)
                    ref_frame = env.envs[0].render_reference()
                    if ref_frame is not None:
                        emit_reference_frame(ref_frame)
                except Exception as e:
                    pass  # Ignore render errors
            
            return True
        
        def _on_training_end(self) -> None:
            # Save final model
            self.model.save(models_dir / "final_model")
            print(f"\n✅ Training complete! Models saved to {models_dir}")
    
    # Create PPO model
    print("🧠 Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(models_dir / "tensorboard"),
    )
    print("✅ PPO agent initialized")
    print()
    
    # Start training
    print("=" * 60)
    print("🚀 Starting training...")
    print(f"📊 Open http://localhost:1306 to view training progress")
    print("=" * 60)
    print()
    
    training_state['is_running'] = True
    
    try:
        model.learn(
            total_timesteps=1_000_000,
            callback=TrainingCallback(render_freq=5),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        model.save(models_dir / "interrupted_model")
        print(f"💾 Model saved to {models_dir / 'interrupted_model'}")
    finally:
        training_state['is_running'] = False
        env.close()
    
    print("\n🏁 Training session ended")
    print(f"📁 Models saved in: {models_dir}")


if __name__ == "__main__":
    main()
