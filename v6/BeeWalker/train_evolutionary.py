"""
BeeWalker Training Script - PARALLEL VERSION
Evolutionary training using all CPU cores for parallel gait evaluation.
Shows the best walker while training runs.
Watch on port 1306.

Each run saves to results/<timestamp>/:
  - config.json: Training configuration and hyperparameters
  - best_params.json: Best gait parameters (weights)
  - training_log.json: Full training history
  - videos/: Video recordings at checkpoints
"""
import os
os.environ['MUJOCO_GL'] = 'egl'

import mujoco
from flask import Flask, Response
import time
import threading
import cv2
import numpy as np
import math
from dataclasses import dataclass, asdict
import copy
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager

app = Flask(__name__)

# Global state (for main process)
latest_frame = None
lock = threading.Lock()
params_lock = threading.Lock()

# Shared state via Manager (for cross-process communication)
manager = None
training_stats = None
current_best_params = None

POPULATION_SIZE = 50  # Increased since we can evaluate faster now!
NUM_WORKERS = None  # Will be set to cpu_count()

# Results directory for this run
RUN_DIR = None

def init_shared_state():
    """Initialize shared state for multiprocessing."""
    global manager, training_stats, current_best_params
    manager = Manager()
    training_stats = manager.dict({"generation": 0, "best_fitness": 0, "evaluating": 0, "workers": 0})
    # We'll use a separate mechanism for current_best_params

def init_run_directory():
    """Create timestamped results directory for this training run."""
    global RUN_DIR
    
    base_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(base_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = os.path.join(base_dir, timestamp)
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(os.path.join(RUN_DIR, "videos"), exist_ok=True)
    
    print(f"Results will be saved to: {RUN_DIR}")
    return RUN_DIR

def save_config(num_workers: int):
    """Save training configuration."""
    config = {
        "population_size": POPULATION_SIZE,
        "num_workers": num_workers,
        "model_file": "model.xml",
        "evaluation_duration": 4.0,
        "timestamp": datetime.now().isoformat(),
        "gait_param_ranges": {
            "gait_frequency": {"min": 0.5, "init_range": [1.0, 2.5]},
            "hip_amplitude": {"min": 0.1, "max": 1.2, "init_range": [0.3, 0.9]},
            "knee_amplitude": {"min": 0.1, "max": 1.2, "init_range": [0.3, 0.9]},
            "ankle_amplitude": {"min": 0.0, "max": 0.8},
            "phase_offset": {"default": math.pi},
            "knee_offset": {"min": -0.5, "max": 0.5}
        }
    }
    
    config_path = os.path.join(RUN_DIR, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

def save_best_params(params: 'GaitParams', fitness: float, generation: int):
    """Save the best gait parameters (model weights)."""
    weights = {
        "generation": generation,
        "fitness": fitness,
        "params": asdict(params),
        "timestamp": datetime.now().isoformat()
    }
    
    weights_path = os.path.join(RUN_DIR, "best_params.json")
    with open(weights_path, 'w') as f:
        json.dump(weights, f, indent=2)

def save_training_log(log_entries: list):
    """Save the full training history."""
    log_path = os.path.join(RUN_DIR, "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(log_entries, f, indent=2)

def record_video(model, params: 'GaitParams', generation: int, duration: float = 5.0):
    """Record a video of the current best walker."""
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    camera = mujoco.MjvCamera()
    camera.distance = 0.8
    camera.elevation = -20
    camera.azimuth = 135
    
    video_path = os.path.join(RUN_DIR, "videos", f"gen_{generation:04d}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    
    sim_time = 0.0
    frame_interval = 1.0 / 30.0  # 30 FPS
    next_frame_time = 0.0
    
    while sim_time < duration:
        apply_gait(data, params, sim_time)
        mujoco.mj_step(model, data)
        sim_time += model.opt.timestep
        
        # Record frame at 30 FPS
        if sim_time >= next_frame_time:
            camera.lookat = data.body("pelvis").xpos.copy()
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render()
            pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            
            # Add stats overlay
            best_fit = training_stats.get('best_fitness', 0) if training_stats else 0
            text = f"Gen {generation} | Fitness: {best_fit:.2f}"
            cv2.putText(pixels_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            video_writer.write(pixels_bgr)
            next_frame_time += frame_interval
    
    video_writer.release()
    print(f"Saved video: {video_path}")

@dataclass
class GaitParams:
    """Parameters that define a walking gait"""
    gait_frequency: float = 1.5
    hip_amplitude: float = 0.5
    knee_amplitude: float = 0.6
    ankle_amplitude: float = 0.3
    phase_offset: float = math.pi
    knee_offset: float = 0.0
    
    def mutate(self, mutation_rate: float = 0.2) -> 'GaitParams':
        new_params = copy.copy(self)
        new_params.gait_frequency = np.clip(self.gait_frequency + np.random.randn() * mutation_rate, 0.5, 3.0)  # Max 3Hz for realistic walking
        new_params.hip_amplitude = np.clip(self.hip_amplitude + np.random.randn() * mutation_rate, 0.1, 1.2)
        new_params.knee_amplitude = np.clip(self.knee_amplitude + np.random.randn() * mutation_rate, 0.1, 1.2)
        new_params.ankle_amplitude = np.clip(self.ankle_amplitude + np.random.randn() * mutation_rate, 0.0, 0.8)
        new_params.phase_offset = self.phase_offset + np.random.randn() * mutation_rate * 0.5
        new_params.knee_offset = np.clip(self.knee_offset + np.random.randn() * mutation_rate * 0.3, -0.5, 0.5)
        return new_params
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: dict) -> 'GaitParams':
        return GaitParams(**d)

def apply_gait(data, params: GaitParams, sim_time: float):
    """Apply gait parameters to control the robot"""
    phase = 2 * math.pi * params.gait_frequency * sim_time
    
    data.ctrl[0] = params.hip_amplitude * math.sin(phase)
    data.ctrl[1] = params.knee_offset + params.knee_amplitude * (math.sin(phase) + 1) / 2
    data.ctrl[2] = params.ankle_amplitude * math.sin(phase + math.pi/4)
    data.ctrl[3] = params.hip_amplitude * math.sin(phase + params.phase_offset)
    data.ctrl[4] = params.knee_offset + params.knee_amplitude * (math.sin(phase + params.phase_offset) + 1) / 2
    data.ctrl[5] = params.ankle_amplitude * math.sin(phase + params.phase_offset + math.pi/4)

def evaluate_gait_worker(params_dict: dict, duration: float = 4.0) -> float:
    """
    Worker function for parallel evaluation.
    Takes a dict (for pickling) and returns fitness score.
    Each worker loads its own model instance.
    
    Improved fitness function with:
    - Frequency penalty for unrealistic speeds
    - Foot alternation reward for actual walking
    - Energy efficiency penalty
    """
    # Load model in worker process (each worker has its own)
    model = mujoco.MjModel.from_xml_path("model.xml")
    data = mujoco.MjData(model)
    
    params = GaitParams.from_dict(params_dict)
    
    start_pos = data.body("pelvis").xpos.copy()
    sim_time = 0.0
    min_height = float('inf')
    total_height = 0.0
    height_samples = 0
    
    # Track joint positions for movement variety
    joint_mins = np.full(6, float('inf'))
    joint_maxs = np.full(6, float('-inf'))
    
    # NEW: Track foot alternation for walking quality
    foot_alternation_sum = 0.0
    total_ctrl_effort = 0.0
    prev_left_foot_z = 0.0
    prev_right_foot_z = 0.0
    step_count = 0
    
    while sim_time < duration:
        apply_gait(data, params, sim_time)
        mujoco.mj_step(model, data)
        sim_time += model.opt.timestep
        
        height = data.body("pelvis").xpos[2]
        min_height = min(min_height, height)
        total_height += height
        height_samples += 1
        
        # Track joint range of motion
        joint_mins = np.minimum(joint_mins, data.ctrl[:6])
        joint_maxs = np.maximum(joint_maxs, data.ctrl[:6])
        
        # NEW: Track foot positions for alternation metric
        left_foot_z = data.body("left_foot").xpos[2]
        right_foot_z = data.body("right_foot").xpos[2]
        foot_alternation_sum += abs(left_foot_z - right_foot_z)
        
        # Detect step transitions (foot going up then down)
        if height_samples > 1:
            if (left_foot_z > prev_left_foot_z + 0.005) or (right_foot_z > prev_right_foot_z + 0.005):
                step_count += 1
        
        prev_left_foot_z = left_foot_z
        prev_right_foot_z = right_foot_z
        
        # NEW: Track control effort for energy penalty
        total_ctrl_effort += np.sum(np.abs(data.ctrl[:6]))
    
    end_pos = data.body("pelvis").xpos.copy()
    
    # Calculate metrics
    forward_distance = end_pos[0] - start_pos[0]
    avg_height = total_height / max(height_samples, 1)
    speed = forward_distance / duration  # meters per second
    
    # Joint activity: how much range each joint uses (0 to ~3 radians max)
    joint_ranges = joint_maxs - joint_mins
    avg_joint_activity = np.mean(joint_ranges)
    
    # NEW: Average foot alternation (higher = more stepping motion)
    avg_foot_alternation = foot_alternation_sum / max(height_samples, 1)
    avg_ctrl_effort = total_ctrl_effort / max(height_samples, 1)
    
    # ===== FITNESS COMPONENTS =====
    
    # Core locomotion rewards
    distance_bonus = forward_distance * 10        # Reward forward distance
    speed_bonus = max(0, speed) * 5               # Reward faster walking
    height_bonus = max(0, avg_height - 0.1) * 2   # Reward staying upright
    activity_bonus = avg_joint_activity * 2       # Reward dynamic movement (reduced weight)
    
    # NEW: Walking quality rewards
    alternation_bonus = avg_foot_alternation * 20  # Strong reward for stepping motion!
    step_bonus = min(step_count * 0.02, 2.0)       # Reward actual steps (capped)
    
    # Penalties
    fall_penalty = -10.0 if min_height < 0.08 else 0  # Increased fall penalty
    lateral_drift = abs(end_pos[1] - start_pos[1])
    drift_penalty = -lateral_drift * 2.0              # Increased drift penalty
    
    # NEW: Frequency penalty - discourage unrealistically fast gaits
    frequency_penalty = max(0, params.gait_frequency - 2.0) * -5.0  # Penalize >2Hz
    
    # NEW: Energy efficiency penalty
    energy_penalty = -avg_ctrl_effort * 0.05  # Small penalty for excessive effort
    
    total_fitness = (
        distance_bonus + speed_bonus + height_bonus + activity_bonus +
        alternation_bonus + step_bonus +
        fall_penalty + drift_penalty + frequency_penalty + energy_penalty
    )
    
    return total_fitness

# Global variable in main process to hold best params for visualization
_viz_best_params = None
_viz_params_lock = threading.Lock()

def training_thread():
    """Background thread that runs parallel evolution"""
    global training_stats, _viz_best_params
    
    print("Starting training thread...")
    
    # Determine number of workers
    num_workers = cpu_count()
    print(f"Using {num_workers} CPU cores for parallel evaluation")
    training_stats["workers"] = num_workers
    
    # Save initial config
    save_config(num_workers)
    
    # Training log
    training_log = []
    
    # Initialize population
    population = []
    for _ in range(POPULATION_SIZE):
        params = GaitParams()
        params.gait_frequency = 1.0 + np.random.rand() * 1.5
        params.hip_amplitude = 0.3 + np.random.rand() * 0.6
        params.knee_amplitude = 0.3 + np.random.rand() * 0.6
        population.append(params)
    
    best_fitness = float('-inf')
    best_params = population[0]
    
    # NEW: Track plateau for diversity injection
    last_improvement_gen = 0
    base_mutation_rate = 0.2
    
    with _viz_params_lock:
        _viz_best_params = copy.copy(best_params)
    
    generation = 0
    last_video_gen = -10  # Record video every 10 generations
    
    # Load model for video recording (in main process)
    model_for_video = mujoco.MjModel.from_xml_path("model.xml")
    
    # Create process pool
    with Pool(processes=num_workers) as pool:
        while True:
            generation += 1
            training_stats["generation"] = generation
            training_stats["evaluating"] = POPULATION_SIZE
            
            gen_start = time.time()
            
            # Parallel evaluation - convert params to dicts for pickling
            params_dicts = [p.to_dict() for p in population]
            fitness_scores = pool.map(evaluate_gait_worker, params_dicts)
            
            gen_time = time.time() - gen_start
            
            # Find best in this generation
            improved = False
            for i, fitness in enumerate(fitness_scores):
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = copy.copy(population[i])
                    training_stats["best_fitness"] = best_fitness
                    last_improvement_gen = generation
                    improved = True
                    
                    with _viz_params_lock:
                        _viz_best_params = copy.copy(best_params)
                    
                    # Save new best parameters
                    save_best_params(best_params, best_fitness, generation)
                    
                    print(f"Gen {generation}: New best! Fitness = {best_fitness:.3f}")
            
            # Log this generation
            gens_since_improvement = generation - last_improvement_gen
            gen_log = {
                "generation": generation,
                "best_fitness": best_fitness,
                "avg_fitness": float(np.mean(fitness_scores)),
                "min_fitness": float(np.min(fitness_scores)),
                "max_fitness": float(np.max(fitness_scores)),
                "eval_time_seconds": gen_time,
                "evals_per_second": POPULATION_SIZE / gen_time,
                "best_params": asdict(best_params),
                "gens_since_improvement": gens_since_improvement,
                "timestamp": datetime.now().isoformat()
            }
            training_log.append(gen_log)
            
            # Save training log every generation
            save_training_log(training_log)
            
            # Record video every 10 generations or on first gen
            if generation == 1 or generation - last_video_gen >= 10:
                record_video(model_for_video, best_params, generation)
                last_video_gen = generation
            
            # Evolution with adaptive mutation
            sorted_indices = np.argsort(fitness_scores)[::-1]
            num_survivors = max(2, POPULATION_SIZE // 3)
            survivors = [population[i] for i in sorted_indices[:num_survivors]]
            
            # NEW: Adaptive mutation rate - increase when stuck
            if gens_since_improvement > 50:
                mutation_rate = min(0.5, base_mutation_rate * (1 + gens_since_improvement / 100))
            else:
                mutation_rate = max(0.05, base_mutation_rate - best_fitness * 0.005)
            
            new_population = [copy.copy(best_params)]
            
            # NEW: Inject random individuals when stuck (10% of population)
            if gens_since_improvement > 100 and gens_since_improvement % 50 == 0:
                print(f"  Injecting diversity (stuck for {gens_since_improvement} gens, mutation={mutation_rate:.3f})")
                for _ in range(POPULATION_SIZE // 10):
                    random_params = GaitParams()
                    random_params.gait_frequency = 0.5 + np.random.rand() * 2.0  # 0.5-2.5 Hz
                    random_params.hip_amplitude = 0.2 + np.random.rand() * 0.8
                    random_params.knee_amplitude = 0.2 + np.random.rand() * 0.8
                    random_params.ankle_amplitude = np.random.rand() * 0.6
                    random_params.phase_offset = math.pi + (np.random.rand() - 0.5) * 0.5
                    random_params.knee_offset = (np.random.rand() - 0.5) * 0.4
                    new_population.append(random_params)
            
            while len(new_population) < POPULATION_SIZE:
                parent = survivors[np.random.randint(len(survivors))]
                new_population.append(parent.mutate(mutation_rate))
            
            population = new_population
            
            avg_fitness = np.mean(fitness_scores)
            print(f"Gen {generation}: Best={best_fitness:.3f}, Avg={avg_fitness:.3f}, "
                  f"Time={gen_time:.2f}s ({POPULATION_SIZE/gen_time:.1f} evals/sec)")

def visualization_thread():
    """Thread that renders the best walker continuously"""
    global latest_frame, _viz_best_params
    
    print("Starting visualization thread...")
    model = mujoco.MjModel.from_xml_path("model.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    camera = mujoco.MjvCamera()
    camera.distance = 0.8
    camera.elevation = -20
    camera.azimuth = 135
    
    sim_time = 0.0
    last_time = time.time()
    
    while True:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        # Get current best params
        with _viz_params_lock:
            params = copy.copy(_viz_best_params) if _viz_best_params else GaitParams()
        
        # Run physics steps to match real time
        steps = int(dt / model.opt.timestep)
        steps = max(1, min(steps, 100))
        
        for _ in range(steps):
            apply_gait(data, params, sim_time)
            mujoco.mj_step(model, data)
            sim_time += model.opt.timestep
        
        # Reset if robot falls or wanders too far
        pelvis_pos = data.body("pelvis").xpos
        if pelvis_pos[2] < 0.05 or abs(pelvis_pos[0]) > 2 or abs(pelvis_pos[1]) > 2:
            mujoco.mj_resetData(model, data)
            sim_time = 0.0
        
        # Camera follows robot
        camera.lookat = data.body("pelvis").xpos.copy()
        
        # Render
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        
        # Stats overlay
        stats = training_stats
        workers = stats.get('workers', 0) if stats else 0
        text1 = f"Gen: {stats['generation']} | Best Fitness: {stats['best_fitness']:.2f}"
        text2 = f"Pop: {POPULATION_SIZE} | Workers: {workers} cores"
        text3 = f"Results: {os.path.basename(RUN_DIR) if RUN_DIR else 'N/A'}"
        cv2.putText(pixels_bgr, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(pixels_bgr, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(pixels_bgr, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(pixels_bgr, "BEST WALKER", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', pixels_bgr)
        if ret:
            with lock:
                latest_frame = buffer.tobytes()
        
        time.sleep(0.033)  # ~30 FPS

def generate_frames():
    global latest_frame
    while True:
        with lock:
            if latest_frame is None:
                time.sleep(0.033)
                continue
            frame = latest_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Initialize shared state for multiprocessing
    init_shared_state()
    
    # Initialize results directory for this run
    init_run_directory()
    
    # Start training in background
    train_t = threading.Thread(target=training_thread, daemon=True)
    train_t.start()
    
    # Start visualization in background
    viz_t = threading.Thread(target=visualization_thread, daemon=True)
    viz_t.start()
    
    print(f"Starting Training Viewer on port 1306...")
    print(f"Using {cpu_count()} CPU cores for parallel evaluation")
    print("Watch your robot learn to walk at http://127.0.0.1:1306")
    app.run(host='0.0.0.0', port=1306, threaded=True)
