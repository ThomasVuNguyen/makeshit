import mujoco
import mujoco.viewer
import time
import math

def main():
    model_path = "model.xml"
    
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
    except ValueError:
        print(f"Error: Could not load model from {model_path}")
        return

    print("Model loaded successfully.")
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")

    # Walking gait parameters
    gait_frequency = 1.5  # Hz - how fast the walking cycle is
    hip_amplitude = 0.5   # radians - how far hips swing
    knee_amplitude = 0.6  # radians - how far knees bend
    ankle_amplitude = 0.3 # radians - ankle adjustment
    
    # Actuator indices (based on model.xml order)
    # 0: left_hip, 1: left_knee, 2: left_ankle
    # 3: right_hip, 4: right_knee, 5: right_ankle

    # Launch the passive viewer
    print("Launching viewer...")
    print("Walking gait engaged!")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            sim_time = time.time() - start_time
            
            # Phase for the walking gait (oscillates 0 to 2*pi)
            phase = 2 * math.pi * gait_frequency * sim_time
            
            # Basic bipedal walking gait:
            # - Left and right legs are 180 degrees out of phase
            # - Hip swings forward/backward
            # - Knee bends during swing phase
            # - Ankle adjusts for ground contact
            
            # Left leg (phase = 0)
            left_hip = hip_amplitude * math.sin(phase)
            left_knee = knee_amplitude * (math.sin(phase) + 1) / 2  # Always bent, more during swing
            left_ankle = ankle_amplitude * math.sin(phase + math.pi/4)
            
            # Right leg (phase = pi, 180 degrees offset)
            right_hip = hip_amplitude * math.sin(phase + math.pi)
            right_knee = knee_amplitude * (math.sin(phase + math.pi) + 1) / 2
            right_ankle = ankle_amplitude * math.sin(phase + math.pi + math.pi/4)
            
            # Apply controls
            data.ctrl[0] = left_hip      # servo_left_hip
            data.ctrl[1] = left_knee     # servo_left_knee
            data.ctrl[2] = left_ankle    # servo_left_ankle
            data.ctrl[3] = right_hip     # servo_right_hip
            data.ctrl[4] = right_knee    # servo_right_knee
            data.ctrl[5] = right_ankle   # servo_right_ankle

            mujoco.mj_step(model, data)
            viewer.sync()

            # Rudimentary time keeping to match real-time roughly
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
