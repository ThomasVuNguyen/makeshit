"""
Flask Web Application for Training Visualization
Real-time training dashboard on port 1306
"""

import threading
import queue
import base64
import io
import os
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO

# Get the directory where this script is located
app_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(app_dir, 'static')

app = Flask(__name__, static_folder=static_dir)
app.config['SECRET_KEY'] = 'beewalker-training-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global metrics queue for thread-safe communication
metrics_queue = queue.Queue()
training_state = {
    "is_running": False,
    "is_paused": False,
    "total_timesteps": 0,
    "episodes": 0,
    "mean_reward": 0,
    "best_reward": float('-inf'),
    "current_frame": None,
}


@app.route('/')
def index():
    """Serve the training dashboard."""
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    socketio.emit('state_update', training_state)


@socketio.on('pause_training')
def handle_pause():
    """Handle pause request."""
    training_state['is_paused'] = True
    socketio.emit('state_update', training_state)


@socketio.on('resume_training')
def handle_resume():
    """Handle resume request."""
    training_state['is_paused'] = False
    socketio.emit('state_update', training_state)


@socketio.on('stop_training')
def handle_stop():
    """Handle stop request."""
    training_state['is_running'] = False
    socketio.emit('state_update', training_state)


def emit_metrics(metrics: dict):
    """
    Emit training metrics to connected clients.
    Called from the training loop.
    """
    # Convert numpy types to native Python types for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if hasattr(value, 'item'):  # numpy scalar
            serializable_metrics[key] = value.item()
        elif isinstance(value, float) and value == float('-inf'):
            serializable_metrics[key] = None
        else:
            serializable_metrics[key] = value
    
    training_state.update(serializable_metrics)
    socketio.emit('metrics_update', training_state)


def emit_frame(frame_data):
    """
    Emit rendered frame to connected clients.
    frame_data should be RGB numpy array.
    """
    if frame_data is not None:
        try:
            from PIL import Image
            img = Image.fromarray(frame_data)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=70)
            frame_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            training_state['current_frame'] = frame_base64
            socketio.emit('frame_update', {'frame': frame_base64})
        except Exception as e:
            pass  # Ignore frame emission errors


def emit_reference_frame(frame_data):
    """
    Emit reference camera frame (fixed angle view of robot).
    """
    if frame_data is not None:
        try:
            from PIL import Image
            img = Image.fromarray(frame_data)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=80)
            frame_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            socketio.emit('reference_frame_update', {'frame': frame_base64})
        except Exception as e:
            pass


def run_server(port=1306):
    """Run the Flask-SocketIO server."""
    print(f"\n🌐 Training UI available at: http://localhost:{port}\n")
    socketio.run(app, host='127.0.0.1', port=port, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)


def start_server_thread(port=1306):
    """Start the server in a background thread."""
    server_thread = threading.Thread(target=run_server, args=(port,), daemon=True)
    server_thread.start()
    return server_thread
