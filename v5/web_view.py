import mujoco
import numpy as np
import threading
import time
import io
from flask import Flask, Response, render_template_string
from PIL import Image

app = Flask(__name__)

# Load model
model_path = "model.xml"
try:
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    print(f"✅ Loaded model from {model_path}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Robot state
running = True
lock = threading.Lock()

# Global renderer removed from here to ensure thread safety
# renderer = mujoco.Renderer(m, height=480, width=640)

def sim_thread():
    """Background thread to step the physics simulation."""
    global running
    print("🚀 Simulation thread started...")
    
    # Control signal freq
    dt = m.opt.timestep
    
    while running:
        start_time = time.time()
        
        with lock:
            # Simple control: Sine wave for testing joints
            t = d.time
            # Apply visible movement to confirm it's working
            # Oscillate all joints slightly
            for i in range(m.nu):
                d.ctrl[i] = 0.5 * np.sin(2 * np.pi * 0.5 * t + i)
                
            mujoco.mj_step(m, d)
            
        # Try to run at roughly real-time (though display loop handles fps)
        elapsed = time.time() - start_time
        if elapsed < dt:
            time.sleep(dt - elapsed)

def generate_frames():
    """Generator function for MJPEG stream."""
    # Initialize renderer in the streaming thread to ensure GL context is valid
    print("🎥 Initializing renderer in streaming thread...")
    local_renderer = mujoco.Renderer(m, height=480, width=640)
    
    while running:
        # Limit framerate for streaming to save bandwidth
        time.sleep(0.05) # ~20 FPS
        
        try:
            with lock:
                local_renderer.update_scene(d, camera="track")
                pixels = local_renderer.render()
            
            img = Image.fromarray(pixels)
            
            # Convert to JPEG byte stream
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
        except Exception as e:
            print(f"Render error: {e}")
            continue

@app.route('/')
def index():
    return render_template_string("""
<html>
  <head>
    <title>BeeWalker Web Viewer</title>
    <style>
      body { background: #1a1a1a; color: #eee; font-family: sans-serif; text-align: center; margin: 0; padding: 20px; }
      h1 { margin-bottom: 20px; color: #ffa500; }
      .container { display: inline-block; border: 2px solid #555; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
      img { display: block; max-width: 100%; height: auto; }
      p { color: #aaa; margin-top: 10px; }
    </style>
  </head>
  <body>
    <h1>🐝 BeeWalker Web Stream</h1>
    <div class="container">
      <img src="{{ url_for('video_feed') }}" width="640" height="480" />
    </div>
    <p>Rendering via MuJoCo + Flask</p>
  </body>
</html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start sim thread
    t = threading.Thread(target=sim_thread, daemon=True)
    t.start()
    
    print("🌐 Starting web server on http://localhost:5000")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("Stopping...")
        running = False
        t.join()
