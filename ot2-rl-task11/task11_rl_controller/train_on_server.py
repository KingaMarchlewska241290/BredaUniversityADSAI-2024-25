"""
Training script for GPU server
Simple, self-contained script that can be run directly on the server
"""

import os
import sys
from datetime import datetime

# Adding parent directory to path:
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from clearml import Task

# Importing wrapper from task11 directory:
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ot2_gym_wrapper import OT2GymWrapper


def make_env():
    def _init():
        env = OT2GymWrapper(
            target_tolerance=0.001,
            max_steps=500,
            render_mode=None  # No GUI on server
        )
        env = Monitor(env)
        return env
    return _init



# CONFIGURATION - MODIFY THESE FOR YOUR EXPERIMENTS

EXPERIMENT_NAME = "rl_full_training_1M"  # Changing this for each run
TOTAL_TIMESTEPS = 1_000_000  # 1 million steps

# Hyperparameters (modify for hyperparameter search):
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
GAMMA = 0.99
NET_ARCH = [256, 256]

# ============================================================================

print("=" * 70)
print("TRAINING RL CONTROLLER ON GPU SERVER")
print("=" * 70)
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Network: {NET_ARCH}")
print("=" * 70)

# Initializing ClearML:
task = Task.init(
    project_name="OT2 RL Controller/Kinga",  # Your name in path
    task_name=EXPERIMENT_NAME,
    tags=["PPO", "GPU", "Full-Training"],
)

# FOR SERVER EXECUTION::
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

config = {
    "learning_rate": LEARNING_RATE,
    "n_steps": N_STEPS,
    "batch_size": BATCH_SIZE,
    "gamma": GAMMA,
    "net_arch": NET_ARCH,
    "total_timesteps": TOTAL_TIMESTEPS,
}
task.connect(config)

print("\nClearML Task initialized!")
print(f"Monitor at: {task.get_output_log_web_page()}")

# Creating directories:
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("best_models", exist_ok=True)

# Creating environments:
print("\nInitializing environments...")
env = DummyVecEnv([make_env()])
eval_env = DummyVecEnv([make_env()])

# Creating callbacks:
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=f"./models/{EXPERIMENT_NAME}/",
    name_prefix="ckpt",
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"./best_models/{EXPERIMENT_NAME}/",
    log_path=f"./logs/{EXPERIMENT_NAME}/",
    eval_freq=5_000,
    n_eval_episodes=10,
    deterministic=True,
)

# Initializing agent:
print("\nInitializing PPO agent...")
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=LEARNING_RATE,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    policy_kwargs=dict(net_arch=NET_ARCH),
    verbose=1,
    tensorboard_log=f"./logs/{EXPERIMENT_NAME}/tensorboard/",
)

print("Agent initialized!")
print(f"\n{model.policy}")

# Training:
print("\n" + "=" * 70)
print("STARTING TRAINING...")
print("=" * 70)
print(f"\nMonitor progress: {task.get_output_log_web_page()}")
print("=" * 70 + "\n")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    
    # Saving final model:
    final_model_path = f"./models/{EXPERIMENT_NAME}/final_model"
    model.save(final_model_path)
    print(f"\nFinal model saved: {final_model_path}")
    
    # Uploading to ClearML:
    task.upload_artifact(
        name='final_model',
        artifact_object=f"{final_model_path}.zip"
    )
    
    print(f"Best model: ./best_models/{EXPERIMENT_NAME}/")
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted.")
    interrupted_path = f"./models/{EXPERIMENT_NAME}/interrupted_model"
    model.save(interrupted_path)
    print(f"Model saved: {interrupted_path}")

finally:
    env.close()
    eval_env.close()
    print("\nTraining session ended.")