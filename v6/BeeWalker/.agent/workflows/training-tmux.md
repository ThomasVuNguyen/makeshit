---
description: how to start LSTM training in a tmux session
---
// turbo-all

1. Kill any existing training process:
```bash
pkill -f "python3 training/train_lstm.py" 2>/dev/null
```

2. Kill any existing tmux session named "train":
```bash
tmux kill-session -t train 2>/dev/null
```

3. Start training in a new tmux session:
```bash
tmux new-session -d -s train "cd /root/makeshit/v6/BeeWalker && source venv/bin/activate && python3 training/train_lstm.py"
```

4. Verify it's running:
```bash
sleep 3 && tmux capture-pane -t train -p | tail -20
```

## Notes
- To attach: `tmux attach -t train`
- To detach: `Ctrl+B, D`
- Dashboard: http://127.0.0.1:1306
- Results saved to: `results/lstm_YYYYMMDD_HHMMSS/`
- To resume from checkpoint: add `--resume path/to/checkpoint.zip` to the python command
