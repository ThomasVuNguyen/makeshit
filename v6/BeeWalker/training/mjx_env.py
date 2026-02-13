"""
BeeWalker MJX Environment — Pure JAX GPU-accelerated environment.

Provides batched simulation of the BeeWalker bipedal robot using MuJoCo XLA (MJX).
All functions are jax.jit-compatible for maximum GPU throughput.

Key differences from CPU bee_walker_env.py:
  - No Python loops — everything is vectorized JAX ops
  - Batched across num_envs using jax.vmap
  - Domain randomization done via JAX random keys
  - Reward function ported to JAX (speed config)
  
Design note: step() does a single MJX physics step (0.002s). The training
loop calls step() multiple times per control step (e.g. 10x = 20ms control).
This keeps the XLA graph small for faster compilation on smaller GPUs.
"""

import os
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


# ============================================================================
# ENV STATE
# ============================================================================

class EnvState(NamedTuple):
    """Complete state for one environment instance."""
    mjx_data: mjx.Data       # MJX physics state
    step_count: jnp.ndarray  # Current control step in episode (scalar int)
    rng: jnp.ndarray         # Per-env PRNG key
    done: jnp.ndarray        # Whether episode is done (scalar bool)
    reward: jnp.ndarray      # Last reward (scalar float)
    # Domain randomization params (set at reset, constant during episode)
    motor_scale: jnp.ndarray    # Motor strength scale factor
    obs_noise_std: jnp.ndarray  # Observation noise std
    curriculum: jnp.ndarray     # Curriculum progress [0, 1]


class EnvInfo(NamedTuple):
    """Per-step info returned alongside obs/reward/done."""
    x_position: jnp.ndarray
    episode_length: jnp.ndarray


# ============================================================================
# ENVIRONMENT 
# ============================================================================

class BeeWalkerMJXEnv:
    """GPU-accelerated BeeWalker environment using MJX.
    
    Usage:
        env = BeeWalkerMJXEnv()
        state = env.reset(jax.random.PRNGKey(0))
        obs = env.get_obs(state)
        state, obs, reward, done, info = env.step(state, action)
        
        # Batched (for training):
        batched_reset = jax.vmap(env.reset)
        batched_step = jax.vmap(env.step)
        keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
        states = batched_reset(keys)
    """
    
    def __init__(self, max_episode_steps: int = 1000):
        self.max_episode_steps = max_episode_steps
        
        # Load model via CPU MuJoCo, then put on device
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "env", "model_mjx.xml"
        )
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mjx_model = mjx.put_model(self.mj_model)
        
        # Cache constant indices (from model structure)
        self.pelvis_id = 1
        self.left_foot_id = 8
        self.right_foot_id = 14
        self.torso_id = 2
        
        # Sensor addresses
        self.accel_adr = 0    # imu_accel: 3 dims
        self.quat_adr = 6     # pelvis_orientation: 4 dims  
        self.angvel_adr = 10  # pelvis_angvel: 3 dims
        
        # Joint indices (after freejoint 7-DOF qpos, 6-DOF qvel)
        self.joint_qpos_idx = jnp.array([7, 8, 9, 10, 11, 12])
        self.joint_qvel_idx = jnp.array([6, 7, 8, 9, 10, 11])
        
        self.obs_dim = 22
        self.act_dim = 6
        
        # Pre-compute reset state on CPU, then transfer to device once
        mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, mj_data)
        mujoco.mj_forward(self.mj_model, mj_data)
        self._init_data = mjx.put_data(self.mj_model, mj_data)
    
    # ------------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------------
    
    def _get_obs(self, state: EnvState) -> jnp.ndarray:
        """Extract 22-dim observation from MJX state.
        
        [quat(4), angvel(3), joint_pos(6), joint_vel(6), accel(3)] = 22
        """
        d = state.mjx_data
        
        quat = d.sensordata[self.quat_adr:self.quat_adr + 4]
        angvel = d.sensordata[self.angvel_adr:self.angvel_adr + 3]
        joint_pos = d.qpos[self.joint_qpos_idx]
        joint_vel = d.qvel[self.joint_qvel_idx]
        accel = d.sensordata[self.accel_adr:self.accel_adr + 3]
        
        obs = jnp.concatenate([quat, angvel, joint_pos, joint_vel, accel])
        
        # Add observation noise
        rng, noise_key = jax.random.split(state.rng)
        noise = jax.random.normal(noise_key, shape=(self.obs_dim,)) * state.obs_noise_std
        obs = obs + noise
        
        return obs
    
    # ------------------------------------------------------------------
    # REWARD (speed config)
    # ------------------------------------------------------------------
    
    def _compute_reward(self, state: EnvState, action: jnp.ndarray) -> jnp.ndarray:
        """Reward computation — speed config from CPU env.
        
        Weights: velocity=5.0, upright=0.3, stepping=1.0, ctrl_cost=0.0001,
                 height=0.5, drift_penalty=0.1, reference_motion=1.0, survival=0.1
        """
        d = state.mjx_data
        
        pelvis_pos = d.xpos[self.pelvis_id]
        pelvis_vel = d.cvel[self.pelvis_id]
        pelvis_mat = d.xmat[self.pelvis_id].reshape(3, 3)
        
        forward_vel = pelvis_vel[3]
        upright = pelvis_mat[2, 2]
        
        # Velocity with standing-still penalty
        velocity_reward = jnp.where(
            jnp.abs(forward_vel) < 0.05, -2.0, forward_vel * 5.0
        )
        
        upright_reward = upright * 0.3
        height_reward = jnp.where(pelvis_pos[2] > 0.15, 0.5, 0.0)
        
        # Foot stepping
        left_foot_z = d.xpos[self.left_foot_id][2]
        right_foot_z = d.xpos[self.right_foot_id][2]
        stepping_reward = jnp.abs(left_foot_z - right_foot_z) * 1.0
        
        ctrl_cost = 0.0001 * jnp.sum(action ** 2)
        drift_penalty = 0.1 * jnp.abs(pelvis_vel[4])
        
        # Reference motion (2Hz gait sine wave)
        phase = (state.step_count / 50.0) * 2.0 * jnp.pi * 2.0
        ref_joints = jnp.array([
            0.4 * jnp.sin(phase),
           -0.3 * jnp.cos(phase),
            0.1 * jnp.sin(phase),
           -0.4 * jnp.sin(phase),
           -0.3 * jnp.cos(phase + jnp.pi),
           -0.1 * jnp.sin(phase),
        ])
        ref_error = jnp.sum((d.qpos[self.joint_qpos_idx] - ref_joints) ** 2)
        reference_reward = jnp.exp(-2.0 * ref_error)
        
        total = (velocity_reward + upright_reward + height_reward +
                 stepping_reward + reference_reward + 0.1 -  # 0.1 = survival
                 ctrl_cost - drift_penalty)
        
        return total
    
    # ------------------------------------------------------------------
    # TERMINATION
    # ------------------------------------------------------------------
    
    def _check_termination(self, state: EnvState) -> jnp.ndarray:
        """Check if episode should terminate (fallen or tilted)."""
        d = state.mjx_data
        pelvis_z = d.xpos[self.pelvis_id][2]
        upright = d.xmat[self.pelvis_id].reshape(3, 3)[2, 2]
        
        tilt_threshold = 0.1 + 0.2 * state.curriculum
        fallen = pelvis_z < 0.08
        tilted = upright < tilt_threshold
        
        return fallen | tilted
    
    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jnp.ndarray) -> EnvState:
        """Reset a single environment."""
        rng, dr_key, noise_key = jax.random.split(rng, 3)
        
        # Start from pre-computed template
        mjx_data = self._init_data
        
        # Tiny random perturbation to avoid identical starts
        qpos_noise = jax.random.uniform(
            noise_key, shape=(self.mj_model.nq,), minval=-0.005, maxval=0.005
        )
        new_qpos = mjx_data.qpos + qpos_noise
        mjx_data = mjx_data.replace(qpos=new_qpos)
        
        # Domain randomization
        dr_keys = jax.random.split(dr_key, 3)
        motor_scale = jax.random.uniform(dr_keys[0], minval=0.9, maxval=1.1)
        obs_noise_std = jnp.float32(0.01)
        
        return EnvState(
            mjx_data=mjx_data,
            step_count=jnp.int32(0),
            rng=rng,
            done=jnp.bool_(False),
            reward=jnp.float32(0.0),
            motor_scale=jnp.float32(motor_scale),
            obs_noise_std=jnp.float32(obs_noise_std),
            curriculum=jnp.float32(0.0),
        )
    
    # ------------------------------------------------------------------
    # STEP (single control step = N_SUBSTEPS physics steps)
    # ------------------------------------------------------------------
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: jnp.ndarray):
        """Advance by one control step (single physics step at 0.02s).
        
        Returns: (new_state, obs, reward, done, info)
        """
        rng, step_key = jax.random.split(state.rng)
        
        # Set control
        action = jnp.clip(action, -1.57, 1.57)
        ctrl = action * state.motor_scale
        d = state.mjx_data.replace(ctrl=ctrl)
        
        # Single physics step (timestep=0.02s = 50Hz control)
        d = mjx.step(self.mjx_model, d)
        
        new_step = state.step_count + 1
        
        new_state = state._replace(
            mjx_data=d,
            step_count=new_step,
            rng=rng,
        )
        
        reward = self._compute_reward(new_state, action)
        terminated = self._check_termination(new_state)
        truncated = new_step >= self.max_episode_steps
        done = terminated | truncated
        
        new_state = new_state._replace(reward=reward, done=done)
        obs = self._get_obs(new_state)
        
        info = EnvInfo(
            x_position=d.xpos[self.pelvis_id][0],
            episode_length=new_step,
        )
        
        return new_state, obs, reward, done, info
    
    def get_obs(self, state: EnvState) -> jnp.ndarray:
        """Public wrapper for getting observation."""
        return self._get_obs(state)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import time
    
    print("Creating env...")
    env = BeeWalkerMJXEnv()
    print("✅ Environment created")
    
    print("Compiling reset (first call is slow — XLA compilation)...")
    t0 = time.time()
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    jax.block_until_ready(state.mjx_data.qpos)
    print(f"✅ Reset compiled in {time.time()-t0:.1f}s")
    
    obs = env.get_obs(state)
    print(f"   obs shape: {obs.shape}")
    
    print("Compiling step...")
    t0 = time.time()
    action = jnp.zeros(6)
    state, obs, reward, done, info = env.step(state, action)
    jax.block_until_ready(obs)
    print(f"✅ Step compiled in {time.time()-t0:.1f}s")
    print(f"   reward: {float(reward):.3f}, done: {bool(done)}")
    
    # Batched
    num_envs = 32
    print(f"\nCompiling batched ops ({num_envs} envs)...")
    t0 = time.time()
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    batched_reset = jax.vmap(env.reset)
    batched_step = jax.vmap(env.step)
    states = batched_reset(keys)
    jax.block_until_ready(states.mjx_data.qpos)
    actions = jnp.zeros((num_envs, 6))
    states, obs, rewards, dones, infos = batched_step(states, actions)
    jax.block_until_ready(obs)
    print(f"✅ Batched compiled in {time.time()-t0:.1f}s")
    
    # Speed test
    print("\n⏱️  Speed test...")
    for _ in range(5):
        states, obs, rewards, dones, infos = batched_step(states, actions)
    jax.block_until_ready(obs)
    
    t0 = time.time()
    for _ in range(200):
        states, obs, rewards, dones, infos = batched_step(states, actions)
    jax.block_until_ready(obs)
    elapsed = time.time() - t0
    
    sps = 200 * num_envs / elapsed
    print(f"✅ Speed: {sps:,.0f} steps/sec")
    print(f"   ({num_envs} envs × 200 steps in {elapsed:.2f}s)")
    print(f"   ~{sps/300:.0f}x faster than CPU (300 sps)")
