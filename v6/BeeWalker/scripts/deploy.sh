#!/bin/bash
# ============================================================
# BeeWalker Remote GPU Training — deploy.sh
# ============================================================
# Upload code to a GPU server, install deps, start MJX training.
#
# Usage:
#   ./deploy.sh user@host              # Deploy & start training
#   ./deploy.sh user@host --logs       # Stream training logs
#   ./deploy.sh user@host --pull       # Pull results back
#   ./deploy.sh user@host --stop       # Stop training
#   ./deploy.sh user@host --status     # Check training status
#
# SSH key options:
#   ./deploy.sh -i ~/.ssh/mykey user@host
#   SSH_KEY=~/.ssh/mykey ./deploy.sh user@host
#
# Custom port:
#   ./deploy.sh -p 2222 user@host
# ============================================================

set -euo pipefail

# ---- Parse args ----
SSH_OPTS=()
ACTION="deploy"

while [[ $# -gt 0 ]]; do
    case $1 in
        --logs)   ACTION="logs"; shift ;;
        --pull)   ACTION="pull"; shift ;;
        --stop)   ACTION="stop"; shift ;;
        --status) ACTION="status"; shift ;;
        -i)       SSH_OPTS+=(-i "$2"); shift 2 ;;
        -p)       SSH_OPTS+=(-p "$2"); shift 2 ;;
        -*)       echo "Unknown flag: $1"; exit 1 ;;
        *)        REMOTE="$1"; shift ;;
    esac
done

if [[ -z "${REMOTE:-}" ]]; then
    echo "Usage: ./deploy.sh [OPTIONS] user@host [--logs|--pull|--stop|--status]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BEEWALKER_DIR="$(dirname "$SCRIPT_DIR")"
REMOTE_DIR="~/BeeWalker"

ssh_cmd() { ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"; }
scp_cmd() { scp "${SSH_OPTS[@]}" "$@"; }
rsync_cmd() { rsync -avz -e "ssh ${SSH_OPTS[*]:-}" "$@"; }

# ============================================================
# ACTIONS
# ============================================================

do_deploy() {
    echo "🚀 BeeWalker Remote GPU Deploy"
    echo "   Target: $REMOTE"
    echo "   Code:   $BEEWALKER_DIR"
    echo ""

    # 1. Upload code
    echo "📦 Uploading BeeWalker code..."
    rsync_cmd \
        --exclude='results/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='.git/' \
        --exclude='venv/' \
        --exclude='*.mp4' \
        --exclude='*.npz' \
        "$BEEWALKER_DIR/" "$REMOTE:$REMOTE_DIR/"
    echo "✅ Code uploaded"

    # 2. Install dependencies
    echo ""
    echo "📥 Installing dependencies on remote..."
    ssh_cmd bash << 'INSTALL_EOF'
set -e
echo "  Checking Python..."
python3 --version

echo "  Installing JAX + MJX..."
pip install --break-system-packages --quiet --upgrade \
    "jax[cuda12]" mujoco-mjx flax optax distrax 2>&1 | tail -5

echo "  ✅ Dependencies installed"

# Verify
python3 -c "
import jax
print(f'  JAX {jax.__version__} — devices: {jax.devices()}')
gpu = any(d.platform == \"gpu\" for d in jax.devices())
print(f'  GPU available: {gpu}')
if not gpu:
    print('  ⚠️  WARNING: No GPU detected! Training will be slow.')
"
INSTALL_EOF

    # 3. Start training
    echo ""
    echo "🏋️ Starting MJX training in tmux..."
    ssh_cmd bash << 'TRAIN_EOF'
set -e
cd ~/BeeWalker

# Kill any existing training
tmux kill-session -t train 2>/dev/null || true

# Start training in tmux
tmux new-session -d -s train \
    "PYTHONUNBUFFERED=1 python3 -u -m training.train_mjx \
        --num-envs 4096 \
        --total-steps 1000000000 \
        --n-steps 256 \
        --save-freq 1000000 \
    2>&1 | tee ~/BeeWalker/training.log"

echo "✅ Training started in tmux session 'train'"
echo "   View logs: ./deploy.sh $USER@$(hostname) --logs"
echo "   Pull results: ./deploy.sh $USER@$(hostname) --pull"
TRAIN_EOF

    echo ""
    echo "✅ Deploy complete! Training is running."
    echo ""
    echo "Next steps:"
    echo "  ./deploy.sh $REMOTE --logs     # Watch live training output"
    echo "  ./deploy.sh $REMOTE --status   # Check if still running"
    echo "  ./deploy.sh $REMOTE --pull     # Download results when done"
    echo "  ./deploy.sh $REMOTE --stop     # Stop training early"
}

do_logs() {
    echo "📺 Streaming training logs (Ctrl+C to detach)..."
    ssh_cmd "tail -f ~/BeeWalker/training.log 2>/dev/null || tmux capture-pane -t train -p -S -50"
}

do_status() {
    echo "📊 Training status:"
    ssh_cmd bash << 'STATUS_EOF'
if tmux has-session -t train 2>/dev/null; then
    echo "  ✅ Training is RUNNING"
    echo ""
    # Show last few log lines
    tail -5 ~/BeeWalker/training.log 2>/dev/null || \
        tmux capture-pane -t train -p -S -5
    echo ""
    # GPU utilization
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
else
    echo "  ⏹️  Training is NOT running"
    # Show last log lines
    echo "  Last output:"
    tail -5 ~/BeeWalker/training.log 2>/dev/null || echo "  (no log file)"
fi
STATUS_EOF
}

do_pull() {
    echo "📥 Pulling results from remote..."
    
    LOCAL_RESULTS="$BEEWALKER_DIR/results/"
    mkdir -p "$LOCAL_RESULTS"
    
    rsync_cmd "$REMOTE:$REMOTE_DIR/results/" "$LOCAL_RESULTS"
    
    echo "✅ Results downloaded to: $LOCAL_RESULTS"
    echo ""
    ls -la "$LOCAL_RESULTS" | tail -10
}

do_stop() {
    echo "⏹️  Stopping training..."
    ssh_cmd "tmux send-keys -t train C-c; sleep 2; tmux kill-session -t train 2>/dev/null || true"
    echo "✅ Training stopped"
}

# ---- Run ----
case $ACTION in
    deploy) do_deploy ;;
    logs)   do_logs ;;
    status) do_status ;;
    pull)   do_pull ;;
    stop)   do_stop ;;
esac
