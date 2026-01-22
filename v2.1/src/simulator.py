"""
MuJoCo simulation launcher and basic RL validation
"""
import os
import numpy as np
import mujoco
import mujoco.viewer


def launch_viewer(mjcf_path: str):
    """Launch MuJoCo viewer with the model"""
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    
    # Launch passive viewer
    mujoco.viewer.launch(model, data)


def run_random_validation(mjcf_path: str, num_steps: int = 1000) -> dict:
    """
    Run random actions to validate model works
    
    Returns dict with validation results
    """
    try:
        model = mujoco.MjModel.from_xml_path(mjcf_path)
        data = mujoco.MjData(model)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load model: {str(e)}"
        }
    
    # Get actuator info
    num_actuators = model.nu
    print(f"Model has {num_actuators} actuators")
    
    # Run random simulation
    positions = []
    for step in range(num_steps):
        # Random control
        if num_actuators > 0:
            data.ctrl[:] = np.random.uniform(-1, 1, num_actuators)
        
        # Step simulation
        try:
            mujoco.mj_step(model, data)
        except Exception as e:
            return {
                "success": False,
                "error": f"Simulation failed at step {step}: {str(e)}"
            }
        
        positions.append(data.qpos.copy() if len(data.qpos) > 0 else [])
    
    # Check for NaN or explosion
    if len(positions) > 0 and len(positions[-1]) > 0:
        final_pos = positions[-1]
        if np.any(np.isnan(final_pos)):
            return {
                "success": False,
                "error": "Simulation produced NaN values"
            }
        if np.any(np.abs(final_pos) > 1000):
            return {
                "success": False,
                "error": "Simulation exploded (positions > 1000)"
            }
    
    return {
        "success": True,
        "num_steps": num_steps,
        "num_actuators": num_actuators,
        "message": "Validation passed"
    }


def run_rl_training(mjcf_path: str, total_timesteps: int = 10000):
    """
    Run a short RL training loop to validate model
    Uses Stable-Baselines3 with PPO
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError as e:
        print(f"RL libraries not available: {e}")
        return None
    
    # Create a simple custom environment wrapper
    class MuJoCoEnv(gym.Env):
        def __init__(self, model_path):
            super().__init__()
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
            
            # Define action and observation spaces
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, 
                shape=(self.model.nu,) if self.model.nu > 0 else (1,),
                dtype=np.float32
            )
            
            obs_dim = self.model.nq + self.model.nv
            obs_dim = max(obs_dim, 1)  # At least 1 dimension
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
            
            self.max_steps = 500
            self.current_step = 0
            
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            mujoco.mj_resetData(self.model, self.data)
            self.current_step = 0
            return self._get_obs(), {}
            
        def step(self, action):
            # Apply action
            if self.model.nu > 0:
                self.data.ctrl[:] = action[:self.model.nu]
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            self.current_step += 1
            
            # Get observation
            obs = self._get_obs()
            
            # Simple reward: negative position norm (encourage staying near origin)
            if len(self.data.qpos) > 0:
                reward = -0.01 * np.sum(self.data.qpos ** 2)
            else:
                reward = 0.0
            
            # Check termination
            terminated = False
            truncated = self.current_step >= self.max_steps
            
            # Check for explosion
            if len(self.data.qpos) > 0 and np.any(np.abs(self.data.qpos) > 100):
                terminated = True
                reward = -100
            
            return obs, reward, terminated, truncated, {}
            
        def _get_obs(self):
            qpos = self.data.qpos.flatten()
            qvel = self.data.qvel.flatten()
            obs = np.concatenate([qpos, qvel]) if len(qpos) > 0 else np.zeros(1)
            return obs.astype(np.float32)
            
        def render(self):
            pass
            
        def close(self):
            pass
    
    print("Creating environment...")
    env = MuJoCoEnv(mjcf_path)
    
    print("Creating PPO model...")
    model = PPO("MlpPolicy", env, verbose=1, n_steps=128, batch_size=64)
    
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    print("Training complete!")
    env.close()
    
    return model
