import os
os.environ['MUJOCO_GL'] = 'egl'
import mujoco
from flask import Flask, Response
import time
import threading
import cv2
import numpy as np
import math

app = Flask(__name__)

# Global state
latest_frame = None
lock = threading.Lock()

def simulation_verified_loop():
    global latest_frame
    
    print("Loading model...")
    try:
        model = mujoco.MjModel.from_xml_path("model.xml")
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=480, width=640)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Setup Camera
    camera = mujoco.MjvCamera()
    camera.lookat = [0, 0, 0.3] # Look at the torso height roughly (updated for smaller robot)
    camera.distance = 0.6
    camera.elevation = -20
    camera.azimuth = 0
    
    print("Simulation loop started.")
    
    start_time = time.time()
    
    while True:
        now = time.time()
        elapsed = now - start_time
        
        # 1. ACTUATION: Wiggle joints
        # Oscillate between -0.5 and 0.5 radians (approx -30 to 30 degrees)
        # Using different speeds for different joints to look "alive"
        trajectory = np.sin(elapsed * 2.0) * 0.5 
        data.ctrl[:] = trajectory
        
        # 2. PHYSICS STEP
        mujoco.mj_step(model, data)
        
        # 3. CAMERA: Auto-Center & Turntable
        # Track the torso position to keep it centered
        camera.lookat = data.body("torso").xpos
        
        # Rotate azimuth slowly
        camera.azimuth = (elapsed * 20) % 360
        
        # 4. RENDER
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        
        pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', pixels_bgr)
        
        if ret:
            with lock:
                latest_frame = buffer.tobytes()
        
        # Target ~60Hz simulation rate roughly
        time.sleep(0.01)

def generate_frames():
    global latest_frame
    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        time.sleep(0.033) # 30 FPS stream

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    t = threading.Thread(target=simulation_verified_loop, daemon=True)
    t.start()
    
    print("Starting Web Viewer on port 5000...")
    app.run(host='0.0.0.0', port=5000, threaded=True)
