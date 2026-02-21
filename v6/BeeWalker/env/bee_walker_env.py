"""
BeeWalker Gymnasium Environment
Custom environment for PPO training with domain randomization from step 0.

Domain randomization (all per-episode, controlled by curriculum):
  - Friction: 0.8-1.2x → 0.3-2.0x
  - Link masses: ±5% → ±20%
  - Motor strength: ±5% → ±15%
  - Gravity tilt: ±0.1 → ±1.0 m/s²
  - Push force: 0.1-0.3N → 0.5-1.5N
  - Observation noise: σ=0.01 → σ=0.05
"""
import os
os.environ['MUJOCO_GL'] = 'egl'

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class BeeWalkerEnv(gym.Env):
    """
    BeeWalker bipedal robot environment with full domain randomization.
    
    Observation space:
        Base (22 dims):
            - Pelvis orientation quaternion (4)
            - Pelvis angular velocity (3)
            - Joint positions (6)
            - Joint velocities (6)
            - IMU accelerometer (3) - from MPU-6050
        Optional phase features (+2 dims):
            - sin(phase), cos(phase)
    
    Action space (6 dims):
        - Joint position commands for hip, knee, ankle (left/right), or
        - Residual joint deltas around a phase-based reference gait
    """
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        render_mode=None,
        max_episode_steps=1000,
        residual_action=False,
        phase_features=False,
        residual_limit=0.35,
        reference_gait_scale=1.0,
        gait_frequency_hz=2.0,
    ):
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
        self._policy_hz = 50.0

        # Control mode
        self._use_residual_action = residual_action
        self._use_phase_features = phase_features
        self._residual_limit = float(residual_limit)
        self._reference_gait_scale = float(reference_gait_scale)
        self._gait_frequency_hz = float(gait_frequency_hz)
        
        # === CURRICULUM ===
        # Single progress variable: 0.0 (easy) → 1.0 (hard)
        # Controls ALL randomization ranges simultaneously
        self._curriculum_progress = 0.0
        
        # === STORE DEFAULTS for domain randomization ===
        self._default_friction = self.model.geom_friction.copy()
        self._default_gravity = self.model.opt.gravity.copy()  # [0, 0, -9.81]
        self._default_mass = self.model.body_mass.copy()
        
        # Per-episode randomization values (set in reset)
        self._motor_strength_scale = 1.0  # Applied in step()
        self._obs_noise_std = 0.01        # Applied in _get_obs()
        self._push_strength = 0.1         # Applied in step()
        self._push_interval = 100         # Applied in step()
        
        # Action space: joint position commands [-1.57, 1.57] rad
        self.action_space = spaces.Box(
            low=-1.57, high=1.57, shape=(6,), dtype=np.float32
        )
        
        # Observation space
        # Base: quat (4) + angvel (3) + qpos joints (6) + qvel joints (6) + accel (3) = 22
        self._base_obs_dim = 22
        self._obs_dim = self._base_obs_dim + (2 if self._use_phase_features else 0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
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
        
        # Body ID for pushes
        self._pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        
        # Sensor indices
        self._accel_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        self._quat_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "pelvis_orientation")
        self._angvel_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "pelvis_angvel")
        
        # Get sensor data addresses
        self._accel_adr = self.model.sensor_adr[self._accel_sensor_id]
        self._quat_adr = self.model.sensor_adr[self._quat_sensor_id]
        self._angvel_adr = self.model.sensor_adr[self._angvel_sensor_id]

    def _get_phase(self):
        return (self._step_count / self._policy_hz) * 2.0 * np.pi * self._gait_frequency_hz

    def _reference_joint_targets(self, phase=None):
        if phase is None:
            phase = self._get_phase()

        base = np.array([
            0.4 * np.sin(phase),
           -0.3 * np.cos(phase),
            0.1 * np.sin(phase),
           -0.4 * np.sin(phase),
           -0.3 * np.cos(phase + np.pi),
           -0.1 * np.sin(phase),
        ], dtype=np.float32)
        return self._reference_gait_scale * base
    
    def _get_obs(self):
        """Construct observation vector with per-step noise."""
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
        
        # === OBSERVATION NOISE (per-step) ===
        # Simulates sensor imperfection — the model can never fully trust its inputs
        if self.np_random is not None and self._obs_noise_std > 0:
            obs += self.np_random.normal(0, self._obs_noise_std, size=obs.shape).astype(np.float32)

        if self._use_phase_features:
            phase = self._get_phase()
            phase_obs = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)
            obs = np.concatenate([obs, phase_obs]).astype(np.float32)
        
        return obs
    
    def _get_info(self):
        pelvis_pos = self.data.body("pelvis").xpos.copy()
        return {
            "x_position": pelvis_pos[0],
            "y_position": pelvis_pos[1],
            "z_position": pelvis_pos[2],
        }
    
    def set_curriculum(self, progress):
        """Set curriculum progress from 0.0 (easy) to 1.0 (hard)."""
        self._curriculum_progress = np.clip(progress, 0.0, 1.0)
    
    def _randomize_domain(self):
        """Apply per-episode domain randomization scaled by curriculum progress.
        
        All randomization starts mild and widens with curriculum:
          progress=0.0: Nearly deterministic (±5% variations)
          progress=1.0: Full randomization (wide ranges)
        """
        p = self._curriculum_progress
        rng = self.np_random
        
        if rng is None:
            return
        
        # --- Friction ---
        # Range widens: [0.95, 1.05] → [0.3, 2.0]
        friction_lo = 1.0 - (0.05 + 0.65 * p)  # 0.95 → 0.3
        friction_hi = 1.0 + (0.05 + 0.95 * p)  # 1.05 → 2.0
        friction_scale = rng.uniform(friction_lo, friction_hi)
        self.model.geom_friction[:] = self._default_friction * friction_scale
        
        # --- Link masses ---
        # Range widens: ±5% → ±20%
        mass_range = 0.05 + 0.15 * p  # 0.05 → 0.20
        for i in range(self.model.nbody):
            if self._default_mass[i] > 0.001:  # Skip massless bodies
                scale = rng.uniform(1.0 - mass_range, 1.0 + mass_range)
                self.model.body_mass[i] = self._default_mass[i] * scale
        
        # --- Motor strength ---
        # Scale factor per episode: easy ±5%, hard ±15%
        motor_range = 0.05 + 0.10 * p  # 0.05 → 0.15
        self._motor_strength_scale = rng.uniform(1.0 - motor_range, 1.0 + motor_range)
        
        # --- Gravity tilt (simulate slopes) ---
        # Easy: ±0.1 m/s² (~0.6°), Hard: ±1.0 m/s² (~5.8°)
        tilt_range = 0.1 + 0.9 * p
        gx = rng.uniform(-tilt_range, tilt_range)
        gy = rng.uniform(-tilt_range, tilt_range)
        self.model.opt.gravity[:] = self._default_gravity + np.array([gx, gy, 0.0])
        
        # --- Observation noise ---
        # Easy: σ=0.01, Hard: σ=0.05
        self._obs_noise_std = 0.01 + 0.04 * p
        
        # --- Push parameters ---
        # Push strength: easy 0.1-0.3N, hard 0.5-1.5N
        push_lo = 0.1 + 0.4 * p   # 0.1 → 0.5
        push_hi = 0.3 + 1.2 * p   # 0.3 → 1.5
        self._push_strength = rng.uniform(push_lo, push_hi)
        
        # Push interval: easy 80-120 steps, hard 25-60 steps
        interval_lo = int(80 - 55 * p)   # 80 → 25
        interval_hi = int(120 - 60 * p)  # 120 → 60
        self._push_interval = rng.integers(max(interval_lo, 25), max(interval_hi, 30) + 1)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        
        # === DOMAIN RANDOMIZATION (per-episode) ===
        self._randomize_domain()
        
        # Small random perturbation to initial state
        if self.np_random is not None:
            noise = self.np_random.uniform(-0.01, 0.01, size=self.model.nq)
            self.data.qpos[:] += noise
        
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        # Apply action with optional residual reference tracking.
        if self._use_residual_action:
            residual = np.clip(action, -self._residual_limit, self._residual_limit)
            ref = self._reference_joint_targets()
            applied_action = np.clip(ref + residual, -1.57, 1.57)
            reward_action = residual
        else:
            applied_action = np.clip(action, -1.57, 1.57)
            reward_action = applied_action

        self.data.ctrl[:6] = applied_action * self._motor_strength_scale
        
        # === PUSH PERTURBATIONS ===
        if self._step_count % self._push_interval == 0 and self.np_random is not None:
            push = self.np_random.uniform(-self._push_strength, self._push_strength, size=3)
            self.data.xfrc_applied[self._pelvis_body_id, :3] = push
        elif self._step_count % self._push_interval == 1:
            self.data.xfrc_applied[self._pelvis_body_id, :3] = 0  # Clear after 1 step
        
        # Step simulation: 10 substeps at 0.002s = 20ms per env step = 50Hz policy
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        self._step_count += 1
        
        # Get observation (with noise)
        obs = self._get_obs()
        
        # Compute reward
        reward = self._compute_reward(reward_action)
        
        # Check termination
        terminated = self._check_termination()
        truncated = self._step_count >= self.max_episode_steps
        
        return obs, reward, terminated, truncated, self._get_info()
    
    def _compute_reward(self, action):
        """
        Natural gait reward — tuned for human-like walking.
        Emphasizes symmetry, smoothness, and efficiency over raw speed.
        """
        pelvis_pos = self.data.body("pelvis").xpos
        pelvis_vel = self.data.body("pelvis").cvel  # [angular, linear]
        
        # Forward velocity reward (reduced weight to avoid exploitation)
        forward_vel = pelvis_vel[3]  # Linear velocity X
        velocity_reward = forward_vel * 1.0
        
        # Standing-still penalty — make standing unprofitable
        if abs(forward_vel) < 0.05:
            velocity_reward = -2.0
        
        # Upright reward - pelvis z-axis should point up
        pelvis_mat = self.data.body("pelvis").xmat.reshape(3, 3)
        upright = pelvis_mat[2, 2]  # Z component of body's Z axis
        upright_reward = upright * 1.0
        
        # Height bonus - encourage staying at good height
        height = pelvis_pos[2]
        height_reward = 0.5 if height > 0.15 else 0.0
        
        # Energy penalty - discourage excessive joint torques
        ctrl_cost = 0.005 * np.sum(action**2)
        
        # Joint velocity penalty - penalize jerky motion
        joint_vel = self.data.qvel[6:12]
        jerk_penalty = 0.005 * np.sum(joint_vel**2)
        
        # Lateral drift penalty
        lateral_vel = abs(pelvis_vel[4])
        drift_penalty = 0.2 * lateral_vel
        
        # Foot alternation bonus — encourage stepping
        left_foot_z = self.data.body("left_foot").xpos[2]
        right_foot_z = self.data.body("right_foot").xpos[2]
        foot_diff = abs(left_foot_z - right_foot_z)
        stepping_bonus = foot_diff * 3.0
        
        # === REFERENCE MOTION REWARD ===
        ref_joints = self._reference_joint_targets()
        joint_pos = np.array([self.data.qpos[i] for i in self._joint_qpos_indices])
        ref_error = np.sum((joint_pos - ref_joints) ** 2)
        reference_reward = 1.5 * np.exp(-2.0 * ref_error)
        
        # Survival bonus
        survival_bonus = 0.1
        
        total_reward = (
            velocity_reward +
            upright_reward +
            height_reward +
            stepping_bonus +
            reference_reward +
            survival_bonus -
            ctrl_cost -
            jerk_penalty -
            drift_penalty
        )
        
        return total_reward
    
    def _check_termination(self):
        """Check if episode should terminate (curriculum-aware)."""
        pelvis_pos = self.data.body("pelvis").xpos
        pelvis_mat = self.data.body("pelvis").xmat.reshape(3, 3)
        upright = pelvis_mat[2, 2]
        
        # Terminate if fallen (too low)
        if pelvis_pos[2] < 0.08:
            return True
        
        # Tilt threshold: lenient early (0.1) → strict later (0.3)
        tilt_threshold = 0.1 + 0.2 * self._curriculum_progress
        if upright < tilt_threshold:
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
            self._cam_target = np.array([0.0, 0.0, 0.25])
        
        # Smooth camera tracking
        pelvis = self.data.body("pelvis").xpos
        smooth = 0.05
        self._cam_target[0] += smooth * (pelvis[0] - self._cam_target[0])
        self._cam_target[1] += smooth * (pelvis[1] - self._cam_target[1])
        self._cam_target[2] = 0.22
        self._camera.lookat[:] = self._cam_target
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
