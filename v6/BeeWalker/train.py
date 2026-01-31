"""
BeeWalker Training Script
Simple evolutionary training with single animated robot view.
Shows the best walker while training runs.
Watch on port 1306.
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
from dataclasses import dataclass
import copy

app = Flask(__name__)

# Global state
latest_frame = None
lock = threading.Lock()
training_stats = {"generation": 0, "best_fitness": 0, "evaluating": 0}
current_best_params = None
params_lock = threading.Lock()

POPULATION_SIZE = 20

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
        new_params.gait_frequency = max(0.5, self.gait_frequency + np.random.randn() * mutation_rate)
        new_params.hip_amplitude = np.clip(self.hip_amplitude + np.random.randn() * mutation_rate, 0.1, 1.2)
        new_params.knee_amplitude = np.clip(self.knee_amplitude + np.random.randn() * mutation_rate, 0.1, 1.2)
        new_params.ankle_amplitude = np.clip(self.ankle_amplitude + np.random.randn() * mutation_rate, 0.0, 0.8)
        new_params.phase_offset = self.phase_offset + np.random.randn() * mutation_rate * 0.5
        new_params.knee_offset = np.clip(self.knee_offset + np.random.randn() * mutation_rate * 0.3, -0.5, 0.5)
        return new_params

def apply_gait(data, params: GaitParams, sim_time: float):
    """Apply gait parameters to control the robot"""
    phase = 2 * math.pi * params.gait_frequency * sim_time
    
    data.ctrl[0] = params.hip_amplitude * math.sin(phase)
    data.ctrl[1] = params.knee_offset + params.knee_amplitude * (math.sin(phase) + 1) / 2
    data.ctrl[2] = params.ankle_amplitude * math.sin(phase + math.pi/4)
    data.ctrl[3] = params.hip_amplitude * math.sin(phase + params.phase_offset)
    data.ctrl[4] = params.knee_offset + params.knee_amplitude * (math.sin(phase + params.phase_offset) + 1) / 2
    data.ctrl[5] = params.ankle_amplitude * math.sin(phase + params.phase_offset + math.pi/4)

def evaluate_gait(model, params: GaitParams, duration: float = 4.0) -> float:
    """Evaluate a gait by simulating and measuring forward progress."""
    data = mujoco.MjData(model)
    
    start_pos = data.body("pelvis").xpos.copy()
    sim_time = 0.0
    min_height = float('inf')
    total_height = 0.0
    height_samples = 0
    
    while sim_time < duration:
        apply_gait(data, params, sim_time)
        mujoco.mj_step(model, data)
        sim_time += model.opt.timestep
        
        height = data.body("pelvis").xpos[2]
        min_height = min(min_height, height)
        total_height += height
        height_samples += 1
    
    end_pos = data.body("pelvis").xpos.copy()
    
    forward_distance = end_pos[0] - start_pos[0]
    avg_height = total_height / max(height_samples, 1)
    height_bonus = max(0, avg_height - 0.1) * 2
    fall_penalty = -5.0 if min_height < 0.08 else 0
    lateral_drift = abs(end_pos[1] - start_pos[1])
    drift_penalty = -lateral_drift * 0.5
    
    return forward_distance * 10 + height_bonus + fall_penalty + drift_penalty

def training_thread():
    """Background thread that runs evolution"""
    global current_best_params, training_stats
    
    print("Starting training thread...")
    model = mujoco.MjModel.from_xml_path("model.xml")
    
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
    
    with params_lock:
        current_best_params = copy.copy(best_params)
    
    generation = 0
    
    while True:
        generation += 1
        training_stats["generation"] = generation
        
        # Evaluate all individuals
        fitness_scores = []
        for i, params in enumerate(population):
            training_stats["evaluating"] = i + 1
            fitness = evaluate_gait(model, params)
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_params = copy.copy(params)
                training_stats["best_fitness"] = best_fitness
                
                with params_lock:
                    current_best_params = copy.copy(best_params)
                
                print(f"Gen {generation}: New best! Fitness = {best_fitness:.3f}")
        
        # Evolution
        sorted_indices = np.argsort(fitness_scores)[::-1]
        num_survivors = max(2, POPULATION_SIZE // 3)
        survivors = [population[i] for i in sorted_indices[:num_survivors]]
        
        new_population = [copy.copy(best_params)]
        while len(new_population) < POPULATION_SIZE:
            parent = survivors[np.random.randint(len(survivors))]
            mutation_rate = max(0.05, 0.3 - best_fitness * 0.01)
            new_population.append(parent.mutate(mutation_rate))
        
        population = new_population
        
        avg_fitness = np.mean(fitness_scores)
        print(f"Gen {generation}: Best={best_fitness:.3f}, Avg={avg_fitness:.3f}")

def visualization_thread():
    """Thread that renders the best walker continuously"""
    global latest_frame, current_best_params
    
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
        with params_lock:
            params = copy.copy(current_best_params) if current_best_params else GaitParams()
        
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
        text1 = f"Gen: {stats['generation']} | Best Fitness: {stats['best_fitness']:.2f}"
        text2 = f"Evaluating: {stats['evaluating']}/{POPULATION_SIZE}"
        cv2.putText(pixels_bgr, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(pixels_bgr, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
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
    # Start training in background
    train_t = threading.Thread(target=training_thread, daemon=True)
    train_t.start()
    
    # Start visualization in background
    viz_t = threading.Thread(target=visualization_thread, daemon=True)
    viz_t.start()
    
    print("Starting Training Viewer on port 1306...")
    print("Watch your robot learn to walk at http://127.0.0.1:1306")
    app.run(host='0.0.0.0', port=1306, threaded=True)
