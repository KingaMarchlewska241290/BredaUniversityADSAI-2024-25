"""
Realistic Local Training - Target: 5mm precision (6 points)
Optimized for MacBook M4 GPU
Expected time: 3-4 hours
"""

import os
import sys
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from clearml import Task

from ot2_gym_wrapper import OT2GymWrapper


def make_env(target_tolerance=0.005):  # 5mm target
    def _init():
        env = OT2GymWrapper(
            target_tolerance=target_tolerance,
            max_steps=300,  # Reduced for faster episodes
            render_mode=None
        )
        env = Monitor(env)
        return env
    return _init


print("=" * 70)
print("LOCAL RL TRAINING - TARGET: 5mm PRECISION (6 POINTS)")
print("=" * 70)
print("Device: MacBook M4 GPU")
print("Expected time: 3-4 hours")
print("Target tolerance: 5mm")
print("=" * 70)

# Initialize ClearML (for tracking only, not remote execution)
task = Task.init(
    project_name="OT2 RL Controller - Local/Kinga",
    task_name="local_training_5mm_target",
    tags=["local", "M4", "5mm-target"],
)

# DON'T execute remotely - run locally
# task.execute_remotely()  ← Commented out!

config = {
    "learning_rate": 1e-3,  # Higher for faster learning
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 5,
    "gamma": 0.95,
    "net_arch": [128, 128],  # Balanced size
    "total_timesteps": 200_000,  # Achievable in 3-4 hours
    "target_tolerance_mm": 5.0,
}
task.connect(config)

# Create directories
os.makedirs("models_local", exist_ok=True)
os.makedirs("logs_local", exist_ok=True)
os.makedirs("best_models_local", exist_ok=True)

# Create environments
print("\nInitializing environments...")
env = DummyVecEnv([make_env(target_tolerance=0.005)])
eval_env = DummyVecEnv([make_env(target_tolerance=0.005)])

# Create callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./models_local/",
    name_prefix="ckpt_5mm",
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_models_local/",
    log_path="./logs_local/",
    eval_freq=5_000,
    n_eval_episodes=10,
    deterministic=True,
)

# Initialize agent
print("\nInitializing PPO agent...")
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    n_steps=512,
    batch_size=128,
    n_epochs=5,
    gamma=0.95,
    policy_kwargs=dict(net_arch=[128, 128]),
    verbose=1,
    tensorboard_log="./logs_local/tensorboard/",
    device="mps",  # Using M4 GPU "mps" for Apple Silicon
)

print(f"Agent initialized on device: {model.device}")
print(f"\n{model.policy}")

# Train
print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print("Training for 200,000 steps")
print("Estimated time: 3-4 hours")
print("Target: < 5mm for 6 points")
print(f"Monitor at: {task.get_output_log_web_page()}")
print("=" * 70 + "\n")

start_time = datetime.now()

try:
    model.learn(
        total_timesteps=200_000,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )
    
    elapsed = datetime.now() - start_time
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Time elapsed: {elapsed}")
    
    # Save final model
    model.save("./models_local/final_model_5mm")
    task.upload_artifact('final_model', './models_local/final_model_5mm.zip')
    
    print("\nModel saved: ./models_local/final_model_5mm.zip")
    
    # IMMEDIATE EVALUATION
    print("\n" + "=" * 70)
    print("EVALUATING FINAL MODEL")
    print("=" * 70)
    
    eval_env_test = OT2GymWrapper(target_tolerance=0.005, max_steps=300, render_mode=None)
    
    errors = []
    successes_5mm = []
    successes_10mm = []
    
    for ep in range(20):  # Test on 20 episodes
        obs, info = eval_env_test.reset()
        done = False
        steps = 0
        
        while not done and steps < 300:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env_test.step(action)
            steps += 1
            done = terminated or truncated
        
        error_mm = info['distance'] * 1000
        errors.append(error_mm)
        successes_5mm.append(error_mm < 5.0)
        successes_10mm.append(error_mm < 10.0)
        
        status = "✓" if error_mm < 5.0 else ("○" if error_mm < 10.0 else "✗")
        print(f"Ep {ep+1:2d}: {error_mm:6.2f}mm {status}")
    
    # Calculate results
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    min_error = np.min(errors)
    max_error = np.max(errors)
    success_5mm = np.mean(successes_5mm) * 100
    success_10mm = np.mean(successes_10mm) * 100
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Episodes: 20")
    print(f"Mean error: {mean_error:.2f}mm")
    print(f"Median error: {median_error:.2f}mm")
    print(f"Min error: {min_error:.2f}mm")
    print(f"Max error: {max_error:.2f}mm")
    print(f"Success rate (<5mm): {success_5mm:.0f}%")
    print(f"Success rate (<10mm): {success_10mm:.0f}%")
    print()
    
    # Determine points
    if mean_error < 1.0:
        points = 8
        print("ACHIEVED < 1mm")
    elif mean_error < 5.0:
        points = 6
        print("ACHIEVED < 5mm")
    elif mean_error < 10.0:
        points = 4
        print("ACHIEVED < 10mm")
    else:
        points = 0
        print("Did not achieve < 10mm")
    
    print(f"\nFINAL SCORE: {points}/8 points")
    print("=" * 70)
    
    # Save results
    import json
    results = {
        'mean_error_mm': float(mean_error),
        'median_error_mm': float(median_error),
        'min_error_mm': float(min_error),
        'max_error_mm': float(max_error),
        'success_rate_5mm': float(success_5mm),
        'success_rate_10mm': float(success_10mm),
        'points': points,
        'training_time': str(elapsed),
        'all_errors': [float(e) for e in errors]
    }
    
    with open('local_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to: local_training_results.json")
    print(f"Best model: ./best_models_local/best_model.zip")
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted!")
    model.save("./models_local/interrupted_model")
    print("Model saved: ./models_local/interrupted_model.zip")

finally:
    env.close()
    eval_env.close()
    eval_env_test.close()