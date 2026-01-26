"""
Render a static preview of the BeeWalker model from multiple angles.
Saves images to be served by the Flask app.
"""

import mujoco
import numpy as np
from pathlib import Path
from PIL import Image
import base64
import io


def render_model_preview(width=400, height=400):
    """Render the BeeWalker model from the tracking camera."""
    model_path = Path(__file__).parent / "beewalker.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    # Reset to initial pose
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    renderer = mujoco.Renderer(model, height=height, width=width)
    
    # Render from the tracking camera
    renderer.update_scene(data, camera="track")
    frame = renderer.render()
    
    renderer.close()
    return frame


def render_custom_view(width=400, height=400, azimuth=135, elevation=-25, distance=1.5):
    """Render the model from a custom camera angle."""
    model_path = Path(__file__).parent / "beewalker.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    # Reset to initial pose
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    renderer = mujoco.Renderer(model, height=height, width=width)
    
    # Create a scene with custom camera settings
    scene = mujoco.MjvScene(model, maxgeom=1000)
    camera = mujoco.MjvCamera()
    
    # Set camera parameters
    camera.lookat[:] = [0, 0, 0.25]  # Look at center of robot
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.distance = distance
    
    option = mujoco.MjvOption()
    
    # Update the scene
    mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    
    # Render using the scene directly
    renderer.update_scene(data, camera="track")  # Use tracking camera as fallback
    frame = renderer.render()
    
    renderer.close()
    return frame


def frame_to_base64(frame):
    """Convert numpy frame to base64 PNG."""
    img = Image.fromarray(frame)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def save_preview_images():
    """Save preview images to the static folder."""
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    
    # Render from tracking camera
    print("Rendering main preview...")
    frame = render_model_preview(600, 600)
    img = Image.fromarray(frame)
    img.save(static_dir / "beewalker_preview.png")
    print(f"Saved: {static_dir / 'beewalker_preview.png'}")
    
    return static_dir / "beewalker_preview.png"


if __name__ == "__main__":
    print("=" * 50)
    print("BeeWalker Model Preview Renderer")
    print("=" * 50)
    save_preview_images()
    print("\nDone!")
