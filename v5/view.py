#!/usr/bin/env python3
"""
BeeWalker MJCF Model Viewer
Native MuJoCo viewer for the BeeWalker robot model.

Usage:
    python view.py              # View default model.xml
    python view.py path/to/model.xml  # View specified model
    
Controls:
    - Left mouse: Rotate view
    - Right mouse: Pan view  
    - Scroll: Zoom in/out
    - Double-click: Reset camera
    - Space: Pause/resume simulation
    - Backspace: Reset simulation
    - Ctrl+Q: Quit
"""

import mujoco
import mujoco.viewer
import numpy as np
import sys
import os
from pathlib import Path

# Default model path
DEFAULT_MODEL = "model.xml"


def print_model_info(model: mujoco.MjModel) -> None:
    """Print summary information about the loaded model."""
    print("\n" + "=" * 50)
    print(f"🐝 BeeWalker Model Viewer")
    print("=" * 50)
    print(f"\n📋 Model Information:")
    print(f"   Name:        {model.names[1:].split(b'\\x00')[0].decode()}")
    print(f"   Bodies:      {model.nbody}")
    print(f"   Joints:      {model.njnt}")
    print(f"   Actuators:   {model.nu}")
    print(f"   Sensors:     {model.nsensor}")
    print(f"   Geometries:  {model.ngeom}")
    
    print(f"\n🦿 Joint Configuration:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "root":
            jnt_range = model.jnt_range[i]
            print(f"   {name}: [{jnt_range[0]:.0f}°, {jnt_range[1]:.0f}°]")
    
    print(f"\n⚡ Actuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            ctrl_range = model.actuator_ctrlrange[i]
            print(f"   {name}: [{ctrl_range[0]:.1f}, {ctrl_range[1]:.1f}]")
    
    print(f"\n📡 Sensors: {model.nsensor}")
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        if name:
            print(f"   - {name}")
    
    print("\n" + "=" * 50)


def print_controls() -> None:
    """Print viewer control instructions."""
    print("\n🎮 Viewer Controls:")
    print("   Left mouse drag:   Rotate view")
    print("   Right mouse drag:  Pan view")
    print("   Scroll wheel:      Zoom")
    print("   Double-click:      Reset camera")
    print("   Space:             Pause/Resume")
    print("   Backspace:         Reset simulation")
    print("   Tab:               Toggle UI")
    print("   Ctrl+Q / Esc:      Quit")
    print("\n   Press any key in viewer for more shortcuts\n")


def load_model(model_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a MuJoCo model from file."""
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        sys.exit(1)
    
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print(f"✅ Model loaded: {model_path}")
        return model, data
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)


def run_passive_viewer(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Run the interactive MuJoCo viewer."""
    print_model_info(model)
    print_controls()
    
    print("🚀 Launching native MuJoCo viewer...")
    print("   (Close the window or press Ctrl+C to exit)\n")
    
    # Launch the native viewer
    mujoco.viewer.launch(model, data)


def run_with_callback(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Run viewer with a control callback for custom behavior."""
    print_model_info(model)
    print_controls()
    
    # Simulation timestep counter
    step_count = [0]
    
    def controller(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Control callback - called at each simulation step."""
        step_count[0] += 1
        
        # Example: Apply sinusoidal motion to demonstrate joint movement
        # Uncomment below to see the robot move:
        # t = data.time
        # for i in range(model.nu):
        #     data.ctrl[i] = 0.3 * np.sin(2 * np.pi * 0.5 * t + i * np.pi / 3)
    
    print("🚀 Launching native MuJoCo viewer with controller...")
    print("   (Close the window or press Ctrl+C to exit)\n")
    
    # Launch with passive viewer and manual stepping
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Apply controller
            controller(model, data)
            
            # Sync viewer
            viewer.sync()


def main():
    """Main entry point."""
    # Determine model path from command line or use default
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Use model.xml in the same directory as this script
        script_dir = Path(__file__).parent
        model_path = str(script_dir / DEFAULT_MODEL)
    
    # Check for special flags
    use_callback = "--callback" in sys.argv or "-c" in sys.argv
    
    # Load the model
    model, data = load_model(model_path)
    
    # Run appropriate viewer mode
    if use_callback:
        run_with_callback(model, data)
    else:
        run_passive_viewer(model, data)
    
    print("\n👋 Viewer closed. Goodbye!")


if __name__ == "__main__":
    main()
