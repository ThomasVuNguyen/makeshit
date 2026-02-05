#!/usr/bin/env python3
"""
Plot training progress from TensorBoard logs.
Creates visualizations of key training metrics.
"""
import os
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Installing tensorboard...")
    import subprocess
    subprocess.run(["pip", "install", "tensorboard"], check=True)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_logs(log_dirs):
    """Load all tensorboard logs from multiple directories."""
    all_data = defaultdict(list)
    
    for log_dir in log_dirs:
        log_dir = Path(log_dir)
        if not log_dir.exists():
            continue
            
        # Find all event files
        event_files = list(log_dir.rglob("events.out.tfevents.*"))
        
        for event_file in event_files:
            try:
                ea = EventAccumulator(str(event_file.parent))
                ea.Reload()
                
                # Get all scalar tags
                tags = ea.Tags().get('scalars', [])
                
                for tag in tags:
                    events = ea.Scalars(tag)
                    for event in events:
                        all_data[tag].append((event.step, event.value))
            except Exception as e:
                print(f"Error loading {event_file}: {e}")
    
    # Sort by step and convert to arrays
    result = {}
    for tag, values in all_data.items():
        values.sort(key=lambda x: x[0])
        steps, vals = zip(*values) if values else ([], [])
        result[tag] = (np.array(steps), np.array(vals))
    
    return result


def smooth(data, window=50):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_training_progress(data, output_path="training_progress.png"):
    """Create training progress plots."""
    
    # Define metrics to plot
    metrics = [
        ('rollout/ep_rew_mean', 'Episode Reward', 'green'),
        ('train/value_loss', 'Value Loss', 'red'),
        ('train/policy_gradient_loss', 'Policy Loss', 'blue'),
        ('train/entropy_loss', 'Entropy Loss', 'purple'),
        ('train/approx_kl', 'Approx KL', 'orange'),
        ('train/clip_fraction', 'Clip Fraction', 'brown'),
    ]
    
    # Filter to available metrics
    available = [(tag, name, color) for tag, name, color in metrics if tag in data]
    
    if not available:
        print("No training metrics found!")
        print(f"Available tags: {list(data.keys())}")
        return
    
    # Create figure
    n_plots = len(available)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]
    
    fig.suptitle('BeeWalker Training Progress', fontsize=16, fontweight='bold')
    
    for ax, (tag, name, color) in zip(axes, available):
        steps, values = data[tag]
        
        # Plot raw data with transparency
        ax.plot(steps / 1e6, values, alpha=0.3, color=color, linewidth=0.5)
        
        # Plot smoothed data
        if len(values) > 50:
            smoothed = smooth(values, 50)
            smooth_steps = steps[:len(smoothed)]
            ax.plot(smooth_steps / 1e6, smoothed, color=color, linewidth=2, label=f'{name} (smoothed)')
        
        ax.set_xlabel('Steps (Millions)')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close()
    
    return output_path


def plot_std_over_time(data, output_path="std_progress.png"):
    """Plot standard deviation over time - key stability metric."""
    
    if 'train/std' not in data:
        print("No std data found")
        return None
    
    steps, std_values = data['train/std']
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(steps / 1e6, std_values, color='purple', alpha=0.5, linewidth=0.5)
    
    if len(std_values) > 50:
        smoothed = smooth(std_values, 50)
        smooth_steps = steps[:len(smoothed)]
        ax.plot(smooth_steps / 1e6, smoothed, color='purple', linewidth=2, label='Std (smoothed)')
    
    # Add horizontal lines for reference
    ax.axhline(y=2.72, color='red', linestyle='--', alpha=0.7, label='Upper clamp (e^1 ≈ 2.72)')
    ax.axhline(y=0.14, color='green', linestyle='--', alpha=0.7, label='Lower clamp (e^-2 ≈ 0.14)')
    
    ax.set_xlabel('Steps (Millions)', fontsize=12)
    ax.set_ylabel('Action Std', fontsize=12)
    ax.set_title('Policy Standard Deviation Over Training\n(Key Stability Metric)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved std plot to: {output_path}")
    plt.close()
    
    return output_path


def main():
    # Find all result directories with tensorboard logs
    results_dir = Path(__file__).parent / "results"
    
    log_dirs = [
        results_dir / "sweep_20260201_170713" / "speed" / "tensorboard",
        results_dir / "speed_continuous_20260202_123818",
        results_dir / "speed_continuous_20260203_105649", 
        results_dir / "speed_continuous_20260203_215409",
    ]
    
    # Also search for any tensorboard subdirectories
    for d in results_dir.iterdir():
        if d.is_dir():
            tb_dir = d / "tensorboard"
            if tb_dir.exists():
                log_dirs.append(tb_dir)
    
    print("Loading tensorboard logs from:")
    for d in log_dirs:
        if d.exists():
            print(f"  ✓ {d}")
    
    # Load all data
    data = load_tensorboard_logs(log_dirs)
    
    print(f"\nFound {len(data)} metrics")
    for tag in sorted(data.keys()):
        steps, _ = data[tag]
        if len(steps) > 0:
            print(f"  {tag}: {len(steps)} points, steps {steps[0]/1e6:.1f}M - {steps[-1]/1e6:.1f}M")
    
    # Create plots
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    plot_training_progress(data, output_dir / "training_progress.png")
    plot_std_over_time(data, output_dir / "std_progress.png")
    
    print(f"\n✅ Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
