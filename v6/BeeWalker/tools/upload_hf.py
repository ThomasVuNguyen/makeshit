#!/usr/bin/env python3
"""
BeeWalker HuggingFace Upload Tool

Uploads training results to HuggingFace Hub with auto-generated README cards.
Each training run gets a README with config, reward curves, and video links.

Usage:
    python tools/upload_hf.py                    # Upload all runs
    python tools/upload_hf.py --run lstm_20260207_013141  # Upload specific run
    python tools/upload_hf.py --latest           # Upload only the latest run
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

REPO_ID = "ThomasTheMaker/BeeWalker-v6"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def get_run_dirs(run_filter=None, latest_only=False):
    """Get list of run directories to upload."""
    if not RESULTS_DIR.exists():
        print(f"❌ Results directory not found: {RESULTS_DIR}")
        return []
    
    runs = sorted([
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "archive"
    ])
    
    if run_filter:
        runs = [r for r in runs if run_filter in r.name]
    
    if latest_only and runs:
        runs = [runs[-1]]
    
    return runs


def parse_run_info(run_dir):
    """Extract metadata from a training run directory."""
    info = {
        "name": run_dir.name,
        "path": str(run_dir),
        "config": {},
        "has_videos": False,
        "has_checkpoints": False,
        "has_reward_log": False,
        "has_progress_plot": False,
        "has_best_model": False,
        "video_count": 0,
        "checkpoint_count": 0,
        "total_steps": 0,
    }
    
    # Load run config
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            info["config"] = json.load(f)
    
    # Check for videos
    video_dir = run_dir / "videos"
    if video_dir.exists():
        videos = list(video_dir.glob("*.mp4"))
        info["has_videos"] = len(videos) > 0
        info["video_count"] = len(videos)
        if videos:
            # Extract max step from video filenames
            steps = []
            for v in videos:
                try:
                    step = int(v.stem.split("_")[-1])
                    steps.append(step)
                except ValueError:
                    pass
            if steps:
                info["total_steps"] = max(steps)
    
    # Check for checkpoints
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("*.zip"))
        info["has_checkpoints"] = len(ckpts) > 0
        info["checkpoint_count"] = len(ckpts)
    
    # Other files
    info["has_reward_log"] = (run_dir / "reward_log.json").exists()
    info["has_progress_plot"] = (run_dir / "progress.png").exists()
    info["has_best_model"] = (run_dir / "best_model.zip").exists()
    
    return info


def format_steps(steps):
    """Format step count nicely: 24890000 → 24.9M"""
    if steps >= 1_000_000:
        return f"{steps / 1_000_000:.1f}M"
    elif steps >= 1_000:
        return f"{steps / 1_000:.0f}K"
    return str(steps)


def generate_run_readme(info):
    """Generate a README.md for a training run."""
    cfg = info["config"]
    name = info["name"]
    steps = format_steps(info["total_steps"])
    
    # Determine run type from name
    if "lstm" in name:
        run_type = "LSTM (RecurrentPPO)"
    elif "speed" in name:
        run_type = "PPO (Speed Config)"
    elif "sweep" in name:
        run_type = "PPO (Multi-Experiment Sweep)"
    else:
        run_type = "PPO"
    
    # Parse timestamp from folder name
    timestamp_str = name.split("_", 1)[-1] if "_" in name else name
    try:
        dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        date_str = dt.strftime("%B %d, %Y at %H:%M")
    except ValueError:
        date_str = timestamp_str
    
    readme = f"""# {name}

**Type:** {run_type}  
**Date:** {date_str}  
**Total Steps:** {steps}  

## Configuration

| Parameter | Value |
|-----------|-------|
"""
    for key, val in cfg.items():
        readme += f"| `{key}` | `{val}` |\n"
    
    readme += "\n## Contents\n\n"
    
    if info["has_best_model"]:
        readme += "- 🏆 **best_model.zip** — Best performing model checkpoint\n"
    if info["has_checkpoints"]:
        readme += f"- 💾 **checkpoints/** — {info['checkpoint_count']} periodic checkpoints\n"
    if info["has_videos"]:
        readme += f"- 🎥 **videos/** — {info['video_count']} training videos\n"
    if info["has_progress_plot"]:
        readme += "- 📊 **progress.png** — Training reward curve\n"
    if info["has_reward_log"]:
        readme += "- 📈 **reward_log.json** — Episode reward history\n"
    
    # Add reward stats if available
    reward_path = Path(info["path"]) / "reward_log.json"
    if reward_path.exists():
        try:
            with open(reward_path) as f:
                rewards = json.load(f)
            if rewards:
                recent = rewards[-min(50, len(rewards)):]
                reward_vals = [r["reward"] for r in recent if "reward" in r]
                if reward_vals:
                    readme += f"""
## Training Stats (Last {len(reward_vals)} Episodes)

| Metric | Value |
|--------|-------|
| Best Reward | {max(r['reward'] for r in rewards if 'reward' in r):.1f} |
| Recent Avg | {sum(reward_vals) / len(reward_vals):.1f} |
| Recent Min | {min(reward_vals):.1f} |
| Recent Max | {max(reward_vals):.1f} |
"""
        except (json.JSONDecodeError, KeyError):
            pass
    
    if info["has_progress_plot"]:
        readme += "\n## Training Curve\n\n![Training Progress](progress.png)\n"
    
    return readme


def generate_repo_readme(all_runs):
    """Generate the top-level README for the HF repo."""
    readme = """# 🐝 BeeWalker v6 — Training Results

Training results for the BeeWalker bipedal robot, using PPO and LSTM-based reinforcement learning in MuJoCo.

**GitHub:** [ThomasVuNguyen/makeshit](https://github.com/ThomasVuNguyen/makeshit)

## Training Runs

| Run | Type | Steps | Videos | Checkpoints |
|-----|------|-------|--------|-------------|
"""
    for info in sorted(all_runs, key=lambda x: x["name"], reverse=True):
        name = info["name"]
        steps = format_steps(info["total_steps"])
        vids = info["video_count"]
        ckpts = info["checkpoint_count"]
        
        if "lstm" in name:
            rtype = "LSTM"
        elif "speed" in name:
            rtype = "PPO-Speed"
        elif "sweep" in name:
            rtype = "Sweep"
        else:
            rtype = "PPO"
        
        readme += f"| [{name}](./{name}/) | {rtype} | {steps} | {vids} | {ckpts} |\n"
    
    readme += """
## Robot

- **Simulation:** MuJoCo (BipedWalker custom env)
- **Servos:** 6× MG996R (hip, knee, ankle × 2)
- **Controller:** RP2040 (target deployment)
- **Algorithm:** RecurrentPPO (LSTM hidden_size=32)

## How to Use

```bash
# Upload results to HuggingFace
python tools/upload_hf.py --latest

# Upload all runs
python tools/upload_hf.py

# Upload specific run
python tools/upload_hf.py --run lstm_20260207_013141
```
"""
    return readme


def upload_to_hf(runs, dry_run=False):
    """Upload run directories to HuggingFace."""
    api = HfApi()
    
    # Check auth
    try:
        api.whoami()
    except Exception:
        print("⚠️  Not logged in. Running `huggingface-cli login`...")
        login()
    
    print(f"\n🚀 Uploading {len(runs)} run(s) to {REPO_ID}\n")
    
    # Collect all run info for top-level README
    all_infos = []
    
    for run_dir in runs:
        info = parse_run_info(run_dir)
        all_infos.append(info)
        
        print(f"📦 {info['name']}")
        print(f"   Steps: {format_steps(info['total_steps'])}")
        print(f"   Videos: {info['video_count']}, Checkpoints: {info['checkpoint_count']}")
        
        # Generate per-run README
        readme_content = generate_run_readme(info)
        readme_path = run_dir / "README.md"
        readme_path.write_text(readme_content)
        print(f"   📝 Generated README.md")
        
        if dry_run:
            print(f"   ⏭️  Dry run — skipping upload")
            continue
        
        # Upload the run directory
        print(f"   ⬆️  Uploading to {REPO_ID}/{info['name']}/...")
        try:
            api.upload_large_folder(
                repo_id=REPO_ID,
                folder_path=str(run_dir),
                path_in_repo=info["name"],
                repo_type="model",
            )
            print(f"   ✅ Done!")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            # Fallback to regular upload
            print(f"   🔄 Retrying with upload_folder...")
            try:
                api.upload_folder(
                    repo_id=REPO_ID,
                    folder_path=str(run_dir),
                    path_in_repo=info["name"],
                    repo_type="model",
                )
                print(f"   ✅ Done (fallback)!")
            except Exception as e2:
                print(f"   ❌ Failed: {e2}")
    
    # Generate and upload top-level README
    if not dry_run and all_infos:
        print(f"\n📝 Generating top-level README...")
        
        # Get info for ALL runs (not just the ones being uploaded)
        all_run_dirs = get_run_dirs()
        all_run_infos = [parse_run_info(d) for d in all_run_dirs]
        
        repo_readme = generate_repo_readme(all_run_infos)
        readme_tmp = RESULTS_DIR / "README.md"
        readme_tmp.write_text(repo_readme)
        
        try:
            api.upload_file(
                path_or_fileobj=str(readme_tmp),
                path_in_repo="README.md",
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"✅ Top-level README uploaded")
        except Exception as e:
            print(f"❌ README upload failed: {e}")
        finally:
            readme_tmp.unlink(missing_ok=True)
    
    print(f"\n🎉 Upload complete!")
    print(f"   View at: https://huggingface.co/{REPO_ID}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload BeeWalker training results to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/upload_hf.py                    # Upload all runs  
  python tools/upload_hf.py --latest           # Upload latest run only
  python tools/upload_hf.py --run lstm_20260207 # Upload matching run(s)
  python tools/upload_hf.py --dry-run          # Preview without uploading
  python tools/upload_hf.py --list             # List available runs
        """
    )
    parser.add_argument("--run", type=str, help="Filter runs by name substring")
    parser.add_argument("--latest", action="store_true", help="Upload only the latest run")
    parser.add_argument("--dry-run", action="store_true", help="Preview without uploading")
    parser.add_argument("--list", action="store_true", help="List available runs and exit")
    args = parser.parse_args()
    
    runs = get_run_dirs(run_filter=args.run, latest_only=args.latest)
    
    if not runs:
        print("❌ No runs found matching criteria")
        print(f"   Looking in: {RESULTS_DIR}")
        return
    
    if args.list:
        print(f"📂 Found {len(runs)} run(s) in {RESULTS_DIR}:\n")
        for run_dir in runs:
            info = parse_run_info(run_dir)
            steps = format_steps(info["total_steps"])
            print(f"  {info['name']:40s}  {steps:>8s}  📹{info['video_count']:>5d}  💾{info['checkpoint_count']:>3d}")
        return
    
    upload_to_hf(runs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
