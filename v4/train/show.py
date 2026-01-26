#!/usr/bin/env python3
"""
BeeWalker Model Viewer
Displays the MuJoCo model in a web browser at localhost:1607
"""

import mujoco
import mujoco.viewer
import numpy as np
import http.server
import socketserver
import threading
import time
import io
from PIL import Image
import base64

PORT = 1608
MODEL_PATH = "beewalker.xml"

# HTML template for the viewer
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>BeeWalker Model Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #fff;
        }
        h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            background: linear-gradient(90deg, #f39c12, #e74c3c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(243, 156, 18, 0.3);
        }
        .subtitle {
            color: #8892b0;
            margin-bottom: 2rem;
        }
        .viewer-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        #model-view {
            border-radius: 8px;
            display: block;
        }
        .controls {
            margin-top: 1.5rem;
            display: flex;
            gap: 1rem;
            justify-content: center;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            color: white;
            font-size: 1rem;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
        }
        .info {
            margin-top: 2rem;
            padding: 1rem 2rem;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            text-align: center;
        }
        .info p { color: #8892b0; margin: 0.5rem 0; }
        .info strong { color: #64ffda; }
    </style>
</head>
<body>
    <h1>🐝 BeeWalker Model Viewer</h1>
    <p class="subtitle">MuJoCo Simulation Preview</p>
    
    <div class="viewer-container">
        <img id="model-view" src="/render" width="800" height="600" alt="Model View">
    </div>
    
    <div class="controls">
        <button onclick="refreshView()">🔄 Refresh View</button>
        <button onclick="rotateView()">↻ Rotate Camera</button>
        <button onclick="openNativeViewer()">🖥️ Open Native Viewer</button>
    </div>
    
    <div class="info">
        <p><strong>Model:</strong> beewalker.xml</p>
        <p><strong>Joints:</strong> 6 (2 legs × 3 joints each)</p>
        <p><strong>Servos:</strong> MG996R (180° range)</p>
    </div>
    
    <script>
        let angle = 0;
        
        function refreshView() {
            document.getElementById('model-view').src = '/render?t=' + Date.now() + '&angle=' + angle;
        }
        
        function rotateView() {
            angle = (angle + 45) % 360;
            refreshView();
        }
        
        function openNativeViewer() {
            fetch('/native').then(() => {
                alert('Native MuJoCo viewer opened! Check your desktop.');
            });
        }
        
        // Auto-refresh every 100ms for smooth animation
        setInterval(refreshView, 100);
    </script>
</body>
</html>
"""

# Global model and data
model = None
data = None
renderer = None

def init_mujoco():
    global model, data, renderer
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=600, width=800)

def render_frame(angle=0):
    global model, data, renderer
    
    # Step simulation slightly for natural pose
    mujoco.mj_step(model, data)
    
    # Create camera and set attributes (MjvCamera doesn't accept kwargs in constructor)
    camera = mujoco.MjvCamera()
    camera.lookat[0] = 0
    camera.lookat[1] = 0
    camera.lookat[2] = 0.2
    camera.distance = 0.8
    camera.azimuth = angle
    camera.elevation = -20
    
    renderer.update_scene(data, camera=camera)
    
    # Render to image
    img = renderer.render()
    
    # Convert to PNG bytes
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return buffer.getvalue()

class ViewerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress logging
    
    def do_GET(self):
        if self.path == '/' or self.path.startswith('/?'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
        
        elif self.path.startswith('/render'):
            # Parse angle from query string
            angle = 0
            if 'angle=' in self.path:
                try:
                    angle = int(self.path.split('angle=')[1].split('&')[0])
                except:
                    pass
            
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            self.wfile.write(render_frame(angle))
        
        elif self.path == '/native':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Opening native viewer...')
            # Launch native viewer in a separate thread
            threading.Thread(target=launch_native_viewer, daemon=True).start()
        
        else:
            self.send_response(404)
            self.end_headers()

def launch_native_viewer():
    """Launch the native MuJoCo viewer"""
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)

def main():
    print(f"\n🐝 BeeWalker Model Viewer")
    print(f"{'='*40}")
    
    print(f"Loading model: {MODEL_PATH}")
    init_mujoco()
    print(f"✓ Model loaded successfully!")
    
    print(f"\n🌐 Starting web server on http://localhost:{PORT}")
    print(f"   Press Ctrl+C to stop\n")
    
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), ViewerHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nShutting down...")

if __name__ == "__main__":
    main()
