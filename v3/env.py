"""
Bipedal Walking Robot Environment
Gymnasium environment for training a bipedal robot to walk
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from pathlib import Path


class BipedalWalkerEnv(gym.Env):
    """Custom Gymnasium environment for bipedal walking."""
    
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Load MuJoCo model
        model_path = Path(__file__).parent / "robot.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        
        # Renderer for visualization
        self.render_mode = render_mode
        self.renderer = None
        if render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        # Action space: 6 motors (left/right hip, knee, ankle)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # Observation space:
        # - 6 joint positions
        # - 6 joint velocities  
        # - 3 torso position (x, y, z)
        # - 4 torso orientation (quaternion)
        # - 3 torso angular velocity
        # - 3 torso linear velocity
        # Total: 6 + 6 + 3 + 4 + 3 + 3 = 25
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )
        
        # Get body ID for torso
        self.torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        
        # Episode parameters
        self.max_steps = 1000
        self.current_step = 0
        self.initial_z = 0.8  # Initial torso height
        
    def _get_obs(self):
        """Get the current observation."""
        # Joint positions (first 6 sensors)
        joint_pos = self.data.sensordata[:6].copy()
        
        # Joint velocities (next 6 sensors)
        joint_vel = self.data.sensordata[6:12].copy()
        
        # Torso position
        torso_pos = self.data.xpos[self.torso_id].copy()
        
        # Torso orientation (quaternion)
        torso_quat = self.data.xquat[self.torso_id].copy()
        
        # Torso velocities (from body)
        torso_vel = self.data.cvel[self.torso_id].copy()  # 6D: angular + linear
        torso_angvel = torso_vel[:3]
        torso_linvel = torso_vel[3:]
        
        return np.concatenate([
            joint_pos,      # 6
            joint_vel,      # 6
            torso_pos,      # 3
            torso_quat,     # 4
            torso_angvel,   # 3
            torso_linvel    # 3
        ]).astype(np.float32)
    
    def _get_reward(self):
        """
        Calculate reward for walking:
        - Forward velocity (positive x direction)
        - Staying upright
        - Energy efficiency
        - Survival bonus
        """
        # Get torso state
        torso_pos = self.data.xpos[self.torso_id]
        torso_vel = self.data.cvel[self.torso_id]
        
        # Forward velocity reward (walking in +x direction)
        forward_vel = torso_vel[3]  # x-velocity
        forward_reward = forward_vel * 2.0
        
        # Height reward - encourage staying at nominal height
        height = torso_pos[2]
        height_reward = -abs(height - self.initial_z) * 2.0
        
        # Upright reward - penalize tilting
        # torso_quat[0] is w component, should be close to 1 when upright
        upright_reward = (self.data.xquat[self.torso_id][0] - 0.5) * 2.0
        
        # Control cost - penalize large actions
        control_cost = -0.01 * np.sum(np.square(self.data.ctrl))
        
        # Survival bonus
        alive_bonus = 0.5
        
        total_reward = forward_reward + height_reward + upright_reward + control_cost + alive_bonus
        
        return total_reward
    
    def _is_terminated(self):
        """Check if episode should terminate (robot fell)."""
        torso_pos = self.data.xpos[self.torso_id]
        
        # Terminate if torso too low (fell down)
        if torso_pos[2] < 0.3:
            return True
        
        # Terminate if torso tilted too much
        # Check if w component of quaternion is too far from 1
        if abs(self.data.xquat[self.torso_id][0]) < 0.5:
            return True
            
        return False
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Add small random perturbation to initial joint positions
        if self.np_random is not None:
            noise = self.np_random.uniform(-0.05, 0.05, size=6)
            # Apply to the 6 actuated joints (skip the first 7 for freejoint)
            self.data.qpos[7:13] = noise
        
        # Run a few simulation steps to settle
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        self.current_step = 0
        
        return self._get_obs(), {}
    
    def step(self, action):
        """Take a step in the environment."""
        # Apply action
        self.data.ctrl[:] = action
        
        # Step simulation
        for _ in range(4):  # 4 substeps for stability
            mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        
        # Get observation and reward
        obs = self._get_obs()
        reward = self._get_reward()
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, {}
    
    def render(self):
        """Render the environment."""
        if self.renderer is None:
            return None
        
        # Update camera to follow robot
        torso_pos = self.data.xpos[self.torso_id]
        self.renderer.update_scene(self.data, camera=mujoco.MjvCamera())
        
        # Set camera to look at robot from the side
        cam = mujoco.MjvCamera()
        cam.lookat[0] = torso_pos[0]
        cam.lookat[1] = torso_pos[1]
        cam.lookat[2] = torso_pos[2]
        cam.distance = 3.0
        cam.azimuth = 90  # Side view
        cam.elevation = -15
        
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render()
    
    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()


# Register the environment
gym.register(
    id="BipedalWalker-v0",
    entry_point="env:BipedalWalkerEnv",
    max_episode_steps=1000,
)
