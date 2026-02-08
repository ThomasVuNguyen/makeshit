"""
BeeWalker Gymnasium Environment
Custom environment for PPO training with IMU sensor feedback.
"""
import os
os.environ['MUJOCO_GL'] = 'egl'

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class BeeWalkerEnv(gym.Env):
    """
    BeeWalker bipedal robot environment.
    
    Observation space (22 dims):
        - Pelvis orientation quaternion (4)
        - Pelvis angular velocity (3)
        - Joint positions (6)
        - Joint velocities (6)
        - IMU accelerometer (3) - from MPU-6050
        - (Note: gyro is redundant with angular velocity)
    
    Action space (6 dims):
        - Joint position commands for hip, knee, ankle (left/right)
    """
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None, max_episode_steps=1000):
        super().__init__()
        
        # Load MuJoCo model
        model_path = os.path.join(os.path.dirname(__file__), "model.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Rendering
        self.render_mode = render_mode
        self._renderer = None
        
        # Episode settings
        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        
        # Action space: joint position commands [-1.57, 1.57] rad
        self.action_space = spaces.Box(
            low=-1.57, high=1.57, shape=(6,), dtype=np.float32
        )
        
        # Observation space
        # Quaternion (4) + angvel (3) + qpos joints (6) + qvel joints (6) + accel (3) = 22
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
        )
        
        # Cache joint indices
        self._joint_names = [
            "left_hip_joint", "left_knee_joint", "left_ankle_joint",
            "right_hip_joint", "right_knee_joint", "right_ankle_joint"
        ]
        self._joint_qpos_indices = []
        self._joint_qvel_indices = []
        for name in self._joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = self.model.jnt_qposadr[joint_id]
            qvel_adr = self.model.jnt_dofadr[joint_id]
            self._joint_qpos_indices.append(qpos_adr)
            self._joint_qvel_indices.append(qvel_adr)
        
        # Body ID for domain randomization (random pushes)
        self._pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        
        # Sensor indices
        self._accel_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        self._quat_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "pelvis_orientation")
        self._angvel_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "pelvis_angvel")
        
        # Get sensor data addresses
        self._accel_adr = self.model.sensor_adr[self._accel_sensor_id]
        self._quat_adr = self.model.sensor_adr[self._quat_sensor_id]
        self._angvel_adr = self.model.sensor_adr[self._angvel_sensor_id]
    
    def _get_obs(self):
        """Construct observation vector."""
        # Pelvis orientation (quaternion from sensor)
        quat = self.data.sensordata[self._quat_adr:self._quat_adr+4].copy()
        
        # Pelvis angular velocity (from sensor)
        angvel = self.data.sensordata[self._angvel_adr:self._angvel_adr+3].copy()
        
        # Joint positions
        joint_pos = np.array([self.data.qpos[i] for i in self._joint_qpos_indices])
        
        # Joint velocities
        joint_vel = np.array([self.data.qvel[i] for i in self._joint_qvel_indices])
        
        # IMU accelerometer
        accel = self.data.sensordata[self._accel_adr:self._accel_adr+3].copy()
        
        obs = np.concatenate([
            quat,       # 4
            angvel,     # 3
            joint_pos,  # 6
            joint_vel,  # 6
            accel,      # 3
        ]).astype(np.float32)
        
        return obs
    
    def _get_info(self):
        pelvis_pos = self.data.body("pelvis").xpos.copy()
        return {
            "x_position": pelvis_pos[0],
            "y_position": pelvis_pos[1],
            "z_position": pelvis_pos[2],
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        
        # Small random perturbation to initial state
        if self.np_random is not None:
            noise = self.np_random.uniform(-0.01, 0.01, size=self.model.nq)
            self.data.qpos[:] += noise
        
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # Apply action (joint position commands)
        action = np.clip(action, -1.57, 1.57)
        self.data.ctrl[:6] = action
        
        # Domain randomization: random push every ~1 second (50 steps at 50Hz)
        if self._step_count % 50 == 0 and self.np_random is not None:
            push = self.np_random.uniform(-0.5, 0.5, size=3)
            self.data.xfrc_applied[self._pelvis_body_id, :3] = push
        elif self._step_count % 50 == 1:
            self.data.xfrc_applied[self._pelvis_body_id, :3] = 0  # Clear after 1 step
        
        # Step simulation: 10 substeps at 0.002s = 20ms per env step = 50Hz policy
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        self._step_count += 1
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward
        reward = self._compute_reward(action)
        
        # Check termination
        terminated = self._check_termination()
        truncated = self._step_count >= self.max_episode_steps
        
        return obs, reward, terminated, truncated, self._get_info()
    
    def _compute_reward(self, action):
        """
        Reward function encouraging forward walking.
        Includes reference motion bonus for faster convergence.
        """
        pelvis_pos = self.data.body("pelvis").xpos
        pelvis_vel = self.data.body("pelvis").cvel  # [angular, linear]
        
        # Forward velocity reward (main objective)
        forward_vel = pelvis_vel[3]  # Linear velocity X
        velocity_reward = forward_vel * 2.0
        
        # Upright reward - pelvis z-axis should point up
        pelvis_mat = self.data.body("pelvis").xmat.reshape(3, 3)
        upright = pelvis_mat[2, 2]  # Z component of body's Z axis
        upright_reward = upright * 0.5
        
        # Height bonus - encourage staying at good height
        height = pelvis_pos[2]
        height_reward = 0.5 if height > 0.15 else 0.0
        
        # Energy penalty - discourage excessive joint torques
        ctrl_cost = 0.001 * np.sum(action**2)
        
        # Lateral drift penalty
        lateral_vel = abs(pelvis_vel[4])  # Linear velocity Y
        drift_penalty = 0.1 * lateral_vel
        
        # Foot alternation bonus (encourage stepping motion)
        left_foot_z = self.data.body("left_foot").xpos[2]
        right_foot_z = self.data.body("right_foot").xpos[2]
        foot_diff = abs(left_foot_z - right_foot_z)
        stepping_bonus = foot_diff * 2.0
        
        # === REFERENCE MOTION REWARD ===
        # Sine-wave walking reference at 2Hz — soft bonus for tracking
        # This bootstraps the search, skipping aimless early phases
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
            stepping_bonus +
            reference_reward +
            survival_bonus -
            ctrl_cost -
            drift_penalty
        )
        
        return total_reward
    
    def _check_termination(self):
        """Check if episode should terminate."""
        pelvis_pos = self.data.body("pelvis").xpos
        pelvis_mat = self.data.body("pelvis").xmat.reshape(3, 3)
        upright = pelvis_mat[2, 2]
        
        # Terminate if fallen (too low or too tilted)
        if pelvis_pos[2] < 0.08:
            return True
        if upright < 0.3:  # More than ~70 degrees tilt
            return True
        
        return False
    
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=720, width=1280)
            self._camera = mujoco.MjvCamera()
            self._camera.distance = 0.8
            self._camera.elevation = -18
            self._camera.azimuth = 130
        
        self._camera.lookat = self.data.body("pelvis").xpos.copy()
        self._renderer.update_scene(self.data, camera=self._camera)
        return self._renderer.render()
    
    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# Register the environment
gym.register(
    id="BeeWalker-v0",
    entry_point="bee_walker_env:BeeWalkerEnv",
    max_episode_steps=1000,
)
