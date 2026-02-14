"""
BeeWalker MJX GPU Training Script
Pure JAX PPO + LSTM training loop for GPU-accelerated training.

Usage:
    python -m training.train_mjx                          # Start fresh (512 envs)
    python -m training.train_mjx --num-envs 256           # Fewer envs (less VRAM)
    python -m training.train_mjx --total-steps 10000000   # Custom total steps
"""

import os
import sys
import time
import json
import argparse
import functools
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Any

os.environ['MUJOCO_GL'] = 'egl'

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import distrax
import numpy as np

# Add project root to path
_BEEWALKER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BEEWALKER_ROOT)

from training.mjx_env import BeeWalkerMJXEnv, EnvState


# ============================================================================
# REFERENCE GAIT
# ============================================================================

def compute_ref_actions(step_counts):
    """Compute reference sine gait actions from step count.
    
    This is the same 2Hz sine wave used in the reward function.
    Returns (batch_size, 6) joint targets.
    """
    phase = (step_counts / 50.0) * 2.0 * jnp.pi * 2.0
    return jnp.stack([
         0.4 * jnp.sin(phase),
        -0.3 * jnp.cos(phase),
         0.1 * jnp.sin(phase),
        -0.4 * jnp.sin(phase),
        -0.3 * jnp.cos(phase + jnp.pi),
        -0.1 * jnp.sin(phase),
    ], axis=-1)


# ============================================================================
# NEURAL NETWORK
# ============================================================================

class LSTMCell(nn.Module):
    hidden_size: int
    @nn.compact
    def __call__(self, carry, x):
        return nn.OptimizedLSTMCell(features=self.hidden_size)(carry, x)


class ActorCritic(nn.Module):
    act_dim: int = 6
    hidden_size: int = 32
    
    @nn.compact
    def __call__(self, obs, lstm_state):
        x = nn.Dense(64)(obs)
        x = nn.tanh(x)
        lstm_state, x = LSTMCell(self.hidden_size)(lstm_state, x)
        
        pi_x = nn.Dense(64)(x)
        pi_x = nn.tanh(pi_x)
        mean = nn.Dense(self.act_dim)(pi_x)
        log_std = self.param('log_std', nn.initializers.zeros, (self.act_dim,))
        
        vf_x = nn.Dense(64)(x)
        vf_x = nn.tanh(vf_x)
        value = nn.Dense(1)(vf_x)
        
        return mean, log_std, value.squeeze(-1), lstm_state
    
    def initial_lstm_state(self):
        return (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))


# ============================================================================
# BEHAVIORAL CLONING PRE-TRAINING
# ============================================================================

def pretrain_bc(params, network, env, num_envs, hidden_size, num_iters=200):
    """Pre-train policy to imitate reference sine gait via behavioral cloning.
    
    Rolls out the reference gait in MJX, collects (obs, ref_action) pairs,
    and trains the policy with MSE loss. This gives the agent approximate
    walking before RL fine-tuning begins.
    """
    print(f"\n{'='*60}")
    print(f"📚 Behavioral Cloning Pre-training ({num_iters} iterations)")
    print(f"{'='*60}")
    
    bc_lr = 1e-3
    bc_opt = optax.adam(bc_lr)
    bc_state = bc_opt.init(params)
    
    batched_reset = jax.vmap(env.reset)
    batched_step = jax.vmap(env.step)
    batched_obs = jax.vmap(env.get_obs)
    
    @jax.jit
    def bc_update(params, bc_state, obs, ref_actions, lstm_h, lstm_c):
        """Single BC gradient step: MSE between policy output and reference."""
        def loss_fn(params):
            def single(o, h, c, ref):
                mean, _, _, (nh, nc) = network.apply({'params': params}, o, (h, c))
                return jnp.mean((mean - ref) ** 2), (nh, nc)
            losses, (new_h, new_c) = jax.vmap(single)(obs, lstm_h, lstm_c, ref_actions)
            return jnp.mean(losses), (new_h, new_c)
        
        (loss, (new_h, new_c)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, bc_state = bc_opt.update(grads, bc_state, params)
        params = optax.apply_updates(params, updates)
        return params, bc_state, loss, new_h, new_c
    
    rng = jax.random.PRNGKey(123)
    t0 = time.time()
    
    for it in range(num_iters):
        # Fresh episodes each iteration
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, num_envs)
        states = batched_reset(keys)
        lstm_h = jnp.zeros((num_envs, hidden_size))
        lstm_c = jnp.zeros((num_envs, hidden_size))
        
        total_loss = 0.0
        n_steps = 64
        
        for t in range(n_steps):
            obs = batched_obs(states)
            ref_actions = compute_ref_actions(states.step_count)
            params, bc_state, loss, lstm_h, lstm_c = bc_update(
                params, bc_state, obs, ref_actions, lstm_h, lstm_c)
            # Step env with reference actions (to get realistic obs)
            states, _, _, _, _ = batched_step(states, ref_actions)
            total_loss += float(loss)
        
        avg_loss = total_loss / n_steps
        if it % 20 == 0 or it == num_iters - 1:
            elapsed = time.time() - t0
            print(f"  BC iter {it:3d}/{num_iters} | loss: {avg_loss:.4f} | time: {elapsed:.1f}s")
    
    elapsed = time.time() - t0
    print(f"\n✅ BC pre-training complete in {elapsed:.1f}s")
    print(f"   Final loss: {avg_loss:.4f}")
    return params


# ============================================================================
# PPO CONFIG
# ============================================================================

class PPOConfig(NamedTuple):
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.025
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: int = 64
    n_minibatches: int = 4
    n_epochs: int = 4
    normalize_advantage: bool = True


# ============================================================================
# PPO ALGORITHM
# ============================================================================

def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """Compute GAE. rewards/dones: (T, N), values: (T+1, N)."""
    def _scan_fn(carry, inp):
        gae = carry
        reward, value, next_value, done = inp
        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        return gae, gae
    
    _, advantages = jax.lax.scan(
        _scan_fn,
        jnp.zeros_like(values[0]),
        (rewards[::-1], values[:-1][::-1], values[1:][::-1], dones[::-1]),
    )
    advantages = advantages[::-1]
    returns = advantages + values[:-1]
    return advantages, returns


def ppo_loss(params, apply_fn, batch, config):
    obs, actions = batch['obs'], batch['action']
    old_log_probs, advantages, returns = batch['log_prob'], batch['advantage'], batch['return']
    lstm_state = (batch['lstm_h'], batch['lstm_c'])
    
    mean, log_std, value, _ = apply_fn({'params': params}, obs, lstm_state)
    std = jnp.exp(log_std)
    dist = distrax.MultivariateNormalDiag(mean, std)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy()
    
    ratio = jnp.exp(new_log_probs - old_log_probs)
    clipped = jnp.clip(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
    policy_loss = -jnp.minimum(ratio * advantages, clipped * advantages).mean()
    value_loss = 0.5 * ((value - returns) ** 2).mean()
    entropy_loss = -entropy.mean()
    
    total = policy_loss + config.vf_coef * value_loss + config.ent_coef * entropy_loss
    return total, {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy': -entropy_loss,
    }


# ============================================================================
# TRAINING
# ============================================================================

def train(
    num_envs: int = 512,
    total_steps: int = 1_000_000_000,
    hidden_size: int = 32,
    lr: float = 3e-4,
    lr_end: float = 3e-5,
    n_steps: int = 64,
    save_freq: int = 500_000,
    resume: str = None,
    pretrain: int = 0,
    **kwargs,
):
    config = PPOConfig(lr=lr, n_steps=n_steps)
    
    # Results dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(_BEEWALKER_ROOT) / f"results/mjx_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    
    with open(run_dir / "run_config.json", 'w') as f:
        json.dump({"num_envs": num_envs, "total_steps": total_steps,
                    "hidden_size": hidden_size, "lr": lr, "lr_end": lr_end,
                    "n_steps": n_steps, "resume": resume}, f, indent=2)
    
    print("=" * 60)
    print("🚀 BeeWalker MJX GPU Training")
    print("=" * 60)
    print(f"  Device:      {jax.devices()[0]}")
    print(f"  Num envs:    {num_envs}")
    print(f"  N steps:     {n_steps}")
    print(f"  Batch size:  {num_envs * n_steps:,}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  LR:          {lr} → {lr_end}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Pre-train:   {'Yes' if pretrain else 'No'}")
    if resume:
        print(f"  Resume from: {resume}")
    print(f"  Results:     {run_dir}")
    print("=" * 60)
    
    # ---- Environment ----
    print("\n⏳ Creating MJX environment...")
    env = BeeWalkerMJXEnv()
    print("✅ Environment created")
    
    # ---- Network ----
    network = ActorCritic(act_dim=6, hidden_size=hidden_size)
    rng = jax.random.PRNGKey(42)
    rng, init_key = jax.random.split(rng)
    params = network.init(init_key, jnp.zeros(env.obs_dim), network.initial_lstm_state())['params']
    total_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"  📊 Model: {total_params:,} params ({total_params * 4 / 1024:.1f} KB)")
    
    # Load checkpoint if resuming
    if resume:
        print(f"  📦 Loading checkpoint: {resume}")
        params = _load_params(resume, params)
        print(f"  ✅ Checkpoint loaded")
    
    # LR schedule: linear decay from lr → lr_end
    num_updates = total_steps // (num_envs * n_steps)
    lr_schedule = optax.linear_schedule(lr, lr_end, num_updates)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=lr_schedule),
    )
    opt_state = optimizer.init(params)
    print(f"  📉 LR schedule: {lr} → {lr_end} over {num_updates:,} updates")
    
    # ---- Pre-train with behavioral cloning ----
    if pretrain and not resume:
        params = pretrain_bc(params, network, env, num_envs, hidden_size, num_iters=pretrain)
        # Re-init optimizer with pre-trained params
        opt_state = optimizer.init(params)
        _save_params(params, run_dir / "pretrained_model.npz")
        print("  💾 Pre-trained params saved")
    
    # ---- Initialize envs ----
    print("⏳ Initializing environments...")
    rng, env_key = jax.random.split(rng)
    env_keys = jax.random.split(env_key, num_envs)
    batched_reset = jax.vmap(env.reset)
    batched_step = jax.vmap(env.step)
    batched_get_obs = jax.vmap(env.get_obs)
    
    env_states = batched_reset(env_keys)
    jax.block_until_ready(env_states.mjx_data.qpos)
    print(f"✅ {num_envs} environments reset on GPU")
    
    # ---- JIT-compiled functions ----
    
    @jax.jit
    def policy_forward(params, obs, lstm_h, lstm_c):
        """Batched forward pass — returns means, log_std, values, new LSTM states."""
        def single(obs_i, h_i, c_i):
            mean, log_std, value, (new_h, new_c) = network.apply(
                {'params': params}, obs_i, (h_i, c_i))
            return mean, log_std, value, new_h, new_c
        return jax.vmap(single)(obs, lstm_h, lstm_c)
    
    @jax.jit
    def auto_reset(env_states, dones, rng):
        """Vectorized auto-reset: reset all done envs in one shot."""
        reset_keys = jax.random.split(rng, num_envs)
        fresh_states = batched_reset(reset_keys)
        
        # Where done, use fresh state; otherwise keep current
        new_states = jax.tree.map(
            lambda curr, fresh: jnp.where(
                # Broadcast done mask to match array shape
                jnp.reshape(dones, (-1,) + (1,) * (curr.ndim - 1)),
                fresh, curr
            ),
            env_states, fresh_states,
        )
        return new_states
    
    @jax.jit
    def ppo_update(params, opt_state, batch, rng):
        """PPO update — multiple epochs × minibatches."""
        batch_size = batch['obs'].shape[0]
        minibatch_size = batch_size // config.n_minibatches
        
        def _epoch(carry, _):
            params, opt_state, rng = carry
            rng, perm_key = jax.random.split(rng)
            perm = jax.random.permutation(perm_key, batch_size)
            
            def _minibatch(carry, start_idx):
                params, opt_state = carry
                idx = jax.lax.dynamic_slice(perm, (start_idx,), (minibatch_size,))
                mb = jax.tree.map(lambda x: x[idx], batch)
                (loss, metrics), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
                    params, network.apply, mb, config)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), metrics
            
            starts = jnp.arange(config.n_minibatches) * minibatch_size
            (params, opt_state), metrics = jax.lax.scan(_minibatch, (params, opt_state), starts)
            return (params, opt_state, rng), metrics
        
        (params, opt_state, rng), all_metrics = jax.lax.scan(
            _epoch, (params, opt_state, rng), None, length=config.n_epochs)
        avg_metrics = jax.tree.map(lambda x: x.mean(), all_metrics)
        return params, opt_state, rng, avg_metrics
    
    # ---- Warmup JIT compilation ----
    print("\n⏳ Compiling JIT functions (one-time cost)...")
    obs = batched_get_obs(env_states)
    lstm_h = jnp.zeros((num_envs, hidden_size))
    lstm_c = jnp.zeros((num_envs, hidden_size))
    
    t0 = time.time()
    means, log_stds, values, new_h, new_c = policy_forward(params, obs, lstm_h, lstm_c)
    jax.block_until_ready(values)
    print(f"  ✅ policy_forward compiled: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    actions = jnp.zeros((num_envs, 6))
    env_states, obs, rewards, dones, infos = batched_step(env_states, actions)
    jax.block_until_ready(obs)
    print(f"  ✅ batched_step compiled: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    rng, reset_rng = jax.random.split(rng)
    env_states = auto_reset(env_states, dones, reset_rng)
    jax.block_until_ready(env_states.mjx_data.qpos)
    print(f"  ✅ auto_reset compiled: {time.time()-t0:.1f}s")
    
    # Re-initialize everything clean
    rng, env_key = jax.random.split(rng)
    env_keys = jax.random.split(env_key, num_envs)
    env_states = batched_reset(env_keys)
    jax.block_until_ready(env_states.mjx_data.qpos)
    obs = batched_get_obs(env_states)
    lstm_h = jnp.zeros((num_envs, hidden_size))
    lstm_c = jnp.zeros((num_envs, hidden_size))
    
    # ---- TRAINING LOOP ----
    print("\n🏃 Starting training...\n")
    
    global_step = 0
    best_reward = float('-inf')
    reward_log = []
    ep_returns = np.zeros(num_envs)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)
    recent_returns = []
    recent_lengths = []
    
    t_start = time.time()
    t_last_log = t_start
    steps_at_last_log = 0
    num_updates = total_steps // (num_envs * n_steps)
    ppo_compiled = False
    
    for update in range(num_updates):
        # ---- COLLECT ROLLOUT ----
        all_obs, all_actions, all_rewards, all_dones = [], [], [], []
        all_log_probs, all_values, all_lstm_h, all_lstm_c = [], [], [], []
        
        for step in range(n_steps):
            all_lstm_h.append(lstm_h)
            all_lstm_c.append(lstm_c)
            all_obs.append(obs)
            
            # Policy forward
            rng, act_key = jax.random.split(rng)
            means, log_stds, values, new_lstm_h, new_lstm_c = policy_forward(
                params, obs, lstm_h, lstm_c)
            
            # Sample actions
            stds_batch = jnp.broadcast_to(jnp.exp(log_stds), means.shape)
            noise = jax.random.normal(act_key, shape=means.shape)
            actions = means + stds_batch * noise
            log_probs = distrax.MultivariateNormalDiag(means, stds_batch).log_prob(actions)
            
            all_actions.append(actions)
            all_values.append(values)
            all_log_probs.append(log_probs)
            
            # Step envs
            env_states, obs, rewards, dones, infos = batched_step(env_states, actions)
            all_rewards.append(rewards)
            all_dones.append(dones)
            
            # Reset LSTM for done envs
            lstm_h = jnp.where(dones[:, None], 0.0, new_lstm_h)
            lstm_c = jnp.where(dones[:, None], 0.0, new_lstm_c)
            
            # Track episodes (on CPU)
            rewards_np = np.asarray(rewards)
            dones_np = np.asarray(dones)
            ep_returns += rewards_np
            ep_lengths += 1
            
            done_mask = dones_np.astype(bool)
            if done_mask.any():
                recent_returns.extend(ep_returns[done_mask].tolist())
                recent_lengths.extend(ep_lengths[done_mask].tolist())
                ep_returns[done_mask] = 0.0
                ep_lengths[done_mask] = 0
                
                # Vectorized auto-reset
                rng, reset_rng = jax.random.split(rng)
                env_states = auto_reset(env_states, dones, reset_rng)
                obs = batched_get_obs(env_states)
        
        global_step += num_envs * n_steps
        
        # ---- COMPUTE ADVANTAGES ----
        _, _, bootstrap_values, _, _ = policy_forward(params, obs, lstm_h, lstm_c)
        
        r = jnp.stack(all_rewards)
        v = jnp.concatenate([jnp.stack(all_values), bootstrap_values[None, :]])
        d = jnp.stack(all_dones).astype(jnp.float32)
        advantages, returns = compute_gae(r, v, d, config.gamma, config.gae_lambda)
        
        # Flatten batch
        batch = {
            'obs': jnp.stack(all_obs).reshape(-1, env.obs_dim),
            'action': jnp.stack(all_actions).reshape(-1, env.act_dim),
            'log_prob': jnp.stack(all_log_probs).reshape(-1),
            'advantage': advantages.reshape(-1),
            'return': returns.reshape(-1),
            'lstm_h': jnp.stack(all_lstm_h).reshape(-1, hidden_size),
            'lstm_c': jnp.stack(all_lstm_c).reshape(-1, hidden_size),
        }
        
        if config.normalize_advantage:
            batch['advantage'] = (batch['advantage'] - batch['advantage'].mean()) / (batch['advantage'].std() + 1e-8)
        
        # ---- PPO UPDATE ----
        if not ppo_compiled:
            t0 = time.time()
        rng, update_key = jax.random.split(rng)
        params, opt_state, _, metrics = ppo_update(params, opt_state, batch, update_key)
        if not ppo_compiled:
            jax.block_until_ready(jax.tree.leaves(params)[0])
            print(f"  ✅ ppo_update compiled: {time.time()-t0:.1f}s")
            ppo_compiled = True
            t_start = time.time()  # Reset timer after compilation
            t_last_log = t_start
            steps_at_last_log = 0
            global_step = 0  # Reset step counter too
        
        # ---- LOGGING (every 5 seconds) ----
        now = time.time()
        if now - t_last_log >= 5.0 or update == 1:
            elapsed = now - t_start
            fps = (global_step - steps_at_last_log) / (now - t_last_log) if now > t_last_log else 0
            
            if recent_returns:
                mean_ret = np.mean(recent_returns[-200:])
                mean_len = np.mean(recent_lengths[-200:])
                
                flag = ""
                if mean_ret > best_reward:
                    best_reward = mean_ret
                    _save_params(params, run_dir / "best_model.npz")
                    flag = " ★ NEW BEST"
                
                reward_log.append({"steps": global_step, "mean_reward": float(mean_ret),
                                     "mean_length": float(mean_len), "fps": float(fps)})
                
                hours = elapsed / 3600
                eta = (total_steps - global_step) / fps / 3600 if fps > 0 else 0
                
                print(f"  Step {global_step:>12,} | "
                      f"R: {mean_ret:>7.1f} | "
                      f"Len: {mean_len:>5.0f} | "
                      f"FPS: {fps:>9,.0f} | "
                      f"Best: {best_reward:>7.1f} | "
                      f"Time: {hours:.1f}h | "
                      f"ETA: {eta:.1f}h{flag}")
            else:
                print(f"  Step {global_step:>12,} | Warming up... | FPS: {fps:>8,.0f}")
            
            t_last_log = now
            steps_at_last_log = global_step
        
        # ---- CHECKPOINT ----
        if global_step % save_freq < (num_envs * n_steps):
            _save_params(params, run_dir / "checkpoints" / f"step_{global_step:09d}.npz")
            with open(run_dir / "reward_log.json", 'w') as f:
                json.dump(reward_log, f)
    
    # ---- FINAL ----
    _save_params(params, run_dir / "final_model.npz")
    with open(run_dir / "reward_log.json", 'w') as f:
        json.dump(reward_log, f)
    
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Total steps:   {global_step:,}")
    print(f"  Total time:    {total_time/3600:.2f} hours")
    print(f"  Avg FPS:       {global_step/total_time:,.0f}")
    print(f"  Best reward:   {best_reward:.1f}")
    print(f"  Results:       {run_dir}")
    print(f"{'='*60}")


def _save_params(params, path):
    flat = {'/'.join(str(k) for k in key): np.array(val) 
            for key, val in jax.tree_util.tree_leaves_with_path(params)}
    np.savez(str(path), **flat)


def _load_params(path, template_params):
    """Load params from npz, reconstructing the tree structure."""
    import re
    data = np.load(str(path), allow_pickle=True)
    params = {}
    for key in data.files:
        clean_key = re.sub(r"\['(.+?)'\]", r"\1", key)
        parts = clean_key.split('/')
        d = params
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = data[key]
    return params


def main():
    parser = argparse.ArgumentParser(description="BeeWalker MJX GPU Training")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--total-steps", type=int, default=1_000_000_000)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-end", type=float, default=3e-5)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--save-freq", type=int, default=500_000)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .npz to resume from")
    parser.add_argument("--pretrain", type=int, default=0,
                        help="BC pre-training iterations (0=disable, 200=recommended)")
    args = parser.parse_args()
    
    train(
        num_envs=args.num_envs,
        total_steps=args.total_steps,
        hidden_size=args.hidden_size,
        lr=args.lr,
        lr_end=args.lr_end,
        n_steps=args.n_steps,
        save_freq=args.save_freq,
        resume=args.resume,
        pretrain=args.pretrain,
    )


if __name__ == "__main__":
    main()
