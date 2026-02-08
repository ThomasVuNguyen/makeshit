# 🐝 BeeWalker

Bipedal walking robot trained with reinforcement learning in MuJoCo.

**Goal:** Rigorously simulate the bipedal robot (6 joints, MG996R servos, RP2040) in MuJoCo, iteratively refining the MJCF model to match the physical hardware's dynamics and control logic — then train a deployable walking policy.

## Structure

```
BeeWalker/
├── env/                       # Core simulation
│   ├── model.xml              # MuJoCo robot model (MJCF)
│   └── bee_walker_env.py      # Gymnasium environment (22-dim obs, 6-dim action)
├── training/                  # Training scripts
│   ├── train_lstm.py          # LSTM training (RecurrentPPO) ← active
│   └── train.py               # Multi-experiment reward sweep
├── tools/                     # Utilities
│   ├── upload_hf.py           # Upload results to HuggingFace
│   ├── simulate.py            # Run a trained model
│   ├── plot_training.py       # Plot training curves
│   └── web_view.py            # Web-based model viewer
├── archive/                   # Old/experimental approaches
├── analysis/                  # Training analysis & docs
│   └── lstm.md                # Walking style evolution analysis
└── results/                   # Training runs (stored on HuggingFace)
```

## Quick Start

```bash
# Train (launches dashboard at :1306)
python3 training/train_lstm.py

# Resume from checkpoint
python3 training/train_lstm.py --resume results/<run>/checkpoints/lstm_500000_steps.zip

# Upload results to HuggingFace
python3 tools/upload_hf.py --latest
```

## Results

Training results are stored on HuggingFace: [ThomasTheMaker/BeeWalker-v6](https://huggingface.co/ThomasTheMaker/BeeWalker-v6)

## Robot Specs

| Component | Details |
|-----------|---------|
| Servos | 6× MG996R (hip, knee, ankle × 2 legs) |
| Controller | RP2040 (Raspberry Pi Pico) |
| Simulation | MuJoCo, 500Hz physics, 50Hz policy |
| Algorithm | RecurrentPPO (LSTM, hidden_size=32) |
| Model Size | ~4KB (deployable on microcontroller) |

## Related

- **documentation/** — Joint & design docs (in repo root)
- **knowledge/** — Interview notes and research
