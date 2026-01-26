"""
BeeWalker Gymnasium Environment
Custom environment for training a bipedal walking robot using MuJoCo.
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path


class BeeWalkerEnv(gym.Env):
    """
    BeeWalker bipedal walking environment.
    
    Observation Space (18 dims):
        - Joint positions (6): hip, knee, ankle for each leg
        - Joint velocities (6): corresponding velocities
        - Torso orientation (3): roll, pitch, yaw
        - Torso angular velocity (3): corresponding rates
    
    Action Space (6 dims):
        - Joint torques normalized to [-1, 1]
    
    Reward:
        - Forward velocity (positive x direction)
        - Upright bonus (torso staying vertical)
        - Energy penalty (minimize control effort)
        - Survival bonus
    """
    
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Load MuJoCo model
        model_path = Path(__file__).parent / "beewalker.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        
        self.render_mode = render_mode
        self.renderer = None
        
        # Simulation parameters
        self.frame_skip = 5
        self.dt = self.model.opt.timestep * self.frame_skip
        
        # Define spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # Observation: joint pos (6) + joint vel (6) + torso orient (3) + torso ang vel (3)
        obs_dim = 18
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Reward weights
        self.forward_weight = 1.0
        self.upright_weight = 0.5
        self.energy_weight = 0.01
        self.survival_bonus = 0.1
        
        # Track metrics for visualization
        self.episode_reward = 0
        self.episode_length = 0
        self.max_episode_steps = 1000
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Add small random noise to initial state
        if self.np_random is not None:
            self.data.qpos[:] += self.np_random.uniform(-0.01, 0.01, self.model.nq)
            self.data.qvel[:] += self.np_random.uniform(-0.01, 0.01, self.model.nv)
        
        mujoco.mj_forward(self.model, self.data)
        
        self.episode_reward = 0
        self.episode_length = 0
        
        return self._get_obs(), {}
    
    def step(self, action):
        # Apply action (scaled to actuator range)
        self.data.ctrl[:] = action
        
        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = self.episode_length >= self.max_episode_steps
        
        self.episode_reward += reward
        self.episode_length += 1
        
        info = {
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length,
            "x_position": self.data.qpos[0],
            "x_velocity": self.data.qvel[0],
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """Construct observation vector."""
        # Joint positions (skip free joint - first 7 qpos values)
        joint_pos = self.data.qpos[7:13].copy()
        
        # Joint velocities (skip free joint - first 6 qvel values)
        joint_vel = self.data.qvel[6:12].copy()
        
        # Torso orientation (quaternion to euler approximation)
        quat = self.data.qpos[3:7]
        # Simple roll, pitch, yaw extraction
        torso_orient = np.array([
            np.arctan2(2*(quat[0]*quat[1] + quat[2]*quat[3]), 1 - 2*(quat[1]**2 + quat[2]**2)),
            np.arcsin(np.clip(2*(quat[0]*quat[2] - quat[3]*quat[1]), -1, 1)),
            np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]**2 + quat[3]**2))
        ])
        
        # Torso angular velocity
        torso_angvel = self.data.qvel[3:6].copy()
        
        return np.concatenate([joint_pos, joint_vel, torso_orient, torso_angvel]).astype(np.float32)
    
    def _compute_reward(self, action):
        """Compute reward for current state."""
        # Forward velocity reward
        x_vel = self.data.qvel[0]
        forward_reward = self.forward_weight * x_vel
        
        # Upright reward (torso z-axis should point up)
        torso_z = self.data.qpos[2]
        upright_reward = self.upright_weight * (torso_z - 0.2)  # Reward for being above ground
        
        # Energy penalty
        energy_penalty = self.energy_weight * np.sum(np.square(action))
        
        # Total reward
        reward = forward_reward + upright_reward - energy_penalty + self.survival_bonus
        
        return reward
    
    def _is_terminated(self):
        """Check if episode should terminate."""
        torso_z = self.data.qpos[2]
        
        # Terminate if robot falls (torso too low)
        if torso_z < 0.1:
            return True
        
        # Terminate if robot tilts too much
        quat = self.data.qpos[3:7]
        pitch = np.arcsin(np.clip(2*(quat[0]*quat[2] - quat[3]*quat[1]), -1, 1))
        if abs(pitch) > np.pi / 3:  # 60 degrees
            return True
        
        return False
    
    def render(self):
        """Render the environment from tracking camera."""
        if self.render_mode == "rgb_array":
            try:
                if self.renderer is None:
                    self.renderer = mujoco.Renderer(self.model, height=400, width=500)
                
                # Use tracking camera
                try:
                    self.renderer.update_scene(self.data, camera="track")
                except Exception:
                    self.renderer.update_scene(self.data)
                
                return self.renderer.render()
            except Exception as e:
                print(f"Render error: {e}")
                return None
        
        return None
    
    def render_reference(self):
        """Render from fixed reference camera for model preview."""
        try:
            if not hasattr(self, 'ref_renderer') or self.ref_renderer is None:
                self.ref_renderer = mujoco.Renderer(self.model, height=300, width=300)
            
            # Use reference camera for fixed view
            try:
                self.ref_renderer.update_scene(self.data, camera="reference")
            except Exception:
                self.ref_renderer.update_scene(self.data, camera="track")
            
            return self.ref_renderer.render()
        except Exception as e:
            return None
    
    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if hasattr(self, 'ref_renderer') and self.ref_renderer is not None:
            self.ref_renderer.close()
            self.ref_renderer = None


# Register the environment
gym.register(
    id="BeeWalker-v1",
    entry_point="env:BeeWalkerEnv",
    max_episode_steps=1000,
)
