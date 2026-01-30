import mujoco
import mujoco.viewer
import time

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

    # Launch the passive viewer
    print("Launching viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()

            # Apply some simple control (e.g., maintain 0 position)
            # In the future, this is where the RP2040 logic will go
            data.ctrl[:] = 0 

            mujoco.mj_step(model, data)
            viewer.sync()

            # Rudimentary time keeping to match real-time roughly
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
