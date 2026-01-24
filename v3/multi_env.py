"""
Multi-Robot Bipedal Walking Environment
10 robots training simultaneously in the same MuJoCo world
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from pathlib import Path


NUM_ROBOTS = 10
JOINTS_PER_ROBOT = 6  # hip, knee, ankle x2


class MultiRobotWalkerEnv(gym.Env):
    """Environment with 10 bipedal robots training together."""
    
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        model_path = Path(__file__).parent / "multi_robot.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        
        self.render_mode = render_mode
        self.renderer = None
        if render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=960)
        
        # Action space: 6 motors per robot × 10 robots = 60
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(JOINTS_PER_ROBOT * NUM_ROBOTS,), dtype=np.float32
        )
        
        # Observation per robot: 6 joint pos + 6 joint vel + 7 body state = 19
        # Total: 19 × 10 = 190
        obs_per_robot = 19
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_per_robot * NUM_ROBOTS,), dtype=np.float32
        )
        
        # Get torso body IDs
        self.torso_ids = []
        for i in range(NUM_ROBOTS):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"torso_{i}")
            self.torso_ids.append(body_id)
        
        self.max_steps = 1000
        self.current_step = 0
        self.initial_z = 0.8
        
    def _get_robot_obs(self, robot_id):
        """Get observation for a single robot."""
        base_joint = robot_id * JOINTS_PER_ROBOT
        
        # Joint positions and velocities (from qpos/qvel, offset by freejoint DOFs)
        # Each robot has 7 DOF freejoint + 6 hinge joints = 13 DOF
        qpos_start = robot_id * 13 + 7  # Skip freejoint qpos (7 values)
        qvel_start = robot_id * 12 + 6  # Skip freejoint qvel (6 values)
        
        joint_pos = self.data.qpos[qpos_start:qpos_start+6].copy()
        joint_vel = self.data.qvel[qvel_start:qvel_start+6].copy()
        
        # Torso state
        torso_id = self.torso_ids[robot_id]
        torso_pos = self.data.xpos[torso_id].copy()
        torso_quat = self.data.xquat[torso_id].copy()
        
        return np.concatenate([joint_pos, joint_vel, torso_pos, torso_quat])
    
    def _get_obs(self):
        """Get observations for all robots."""
        obs = []
        for i in range(NUM_ROBOTS):
            obs.append(self._get_robot_obs(i))
        return np.concatenate(obs).astype(np.float32)
    
    def _get_robot_reward(self, robot_id):
        """Calculate reward for a single robot."""
        torso_id = self.torso_ids[robot_id]
        torso_pos = self.data.xpos[torso_id]
        torso_quat = self.data.xquat[torso_id]
        
        # Forward velocity (x direction)
        cvel = self.data.cvel[torso_id]
        forward_vel = cvel[3]
        forward_reward = forward_vel * 2.0
        
        # Stay upright
        height = torso_pos[2]
        height_reward = -abs(height - self.initial_z) * 2.0
        upright_reward = (torso_quat[0] - 0.5) * 2.0
        
        # Control cost
        base_ctrl = robot_id * JOINTS_PER_ROBOT
        ctrl = self.data.ctrl[base_ctrl:base_ctrl+JOINTS_PER_ROBOT]
        control_cost = -0.01 * np.sum(np.square(ctrl))
        
        alive_bonus = 0.5
        
        return forward_reward + height_reward + upright_reward + control_cost + alive_bonus
    
    def _get_reward(self):
        """Get total reward (sum of all robots)."""
        total = 0
        for i in range(NUM_ROBOTS):
            total += self._get_robot_reward(i)
        return total
    
    def _is_any_terminated(self):
        """Check if any robot has fallen."""
        for i in range(NUM_ROBOTS):
            torso_pos = self.data.xpos[self.torso_ids[i]]
            torso_quat = self.data.xquat[self.torso_ids[i]]
            
            if torso_pos[2] < 0.3 or abs(torso_quat[0]) < 0.3:
                return True
        return False
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Add small noise to initial positions
        if self.np_random is not None:
            for i in range(NUM_ROBOTS):
                qpos_start = i * 13 + 7
                noise = self.np_random.uniform(-0.05, 0.05, size=6)
                self.data.qpos[qpos_start:qpos_start+6] = noise
        
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        self.current_step = 0
        return self._get_obs(), {}
    
    def step(self, action):
        self.data.ctrl[:] = action
        
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_any_terminated()
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, {}
    
    def render(self):
        if self.renderer is None:
            return None
        
        # Wide camera view to see all 10 robots from the side
        cam = mujoco.MjvCamera()
        cam.lookat[0] = 0
        cam.lookat[1] = 0
        cam.lookat[2] = 0.5
        cam.distance = 15
        cam.azimuth = 90   # Side view to see robots in a row
        cam.elevation = -10
        
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render()
    
    def close(self):
        if self.renderer is not None:
            self.renderer.close()


gym.register(
    id="MultiRobotWalker-v0",
    entry_point="multi_env:MultiRobotWalkerEnv",
    max_episode_steps=1000,
)
