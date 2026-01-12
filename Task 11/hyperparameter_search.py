"""
Hyperparameter Search for RL Controller
Runs multiple training sessions with different hyperparameter configurations
Each group member tests assigned configurations and compares results
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from clearml import Task

from ot2_gym_wrapper import OT2GymWrapper


def make_env():
    def _init():
        env = OT2GymWrapper(
            target_tolerance=0.001,
            max_steps=500,
            render_mode=None
        )
        env = Monitor(env)
        return env
    return _init


def train_with_config(config_name, config, total_timesteps=500_000):
    """
    Training a single configuration
    
    Args:
        config_name: Name for this experiment
        config: Dictionary of hyperparameters
        total_timesteps: How long to train
        
    Returns:
        Final evaluation results
    """
    print("\n" + "=" * 70)
    print(f"TRAINING: {config_name}")
    print("=" * 70)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 70)
    
    # Initialize ClearML
    task = Task.init(
        project_name="OT2 RL Controller - Hyperparameter Search/Kinga",  # Add your name
        task_name=config_name,
        tags=["hyperparameter-search", config.get('algorithm', 'PPO')],
    )

    task.set_base_docker('deanis/2023y2b-rl:latest')
    task.execute_remotely(queue_name="default")

    task.connect(config)
    
    # Creating directories:
    os.makedirs(f"models/{config_name}", exist_ok=True)
    os.makedirs(f"best_models/{config_name}", exist_ok=True)
    os.makedirs(f"logs/{config_name}", exist_ok=True)

    # Creating environments:
    env = DummyVecEnv([make_env()])
    eval_env = DummyVecEnv([make_env()])

    # Creating callbacks:
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=f"./models/{config_name}/",
        name_prefix="ckpt",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./best_models/{config_name}/",
        log_path=f"./logs/{config_name}/",
        eval_freq=5_000,
        n_eval_episodes=10,
        deterministic=True,
    )
    
    # Initializing agent:
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_range=config.get('clip_range', 0.2),
        policy_kwargs=dict(net_arch=config['net_arch']),
        verbose=1,
        tensorboard_log=f"./logs/{config_name}/tensorboard/",
    )
    
    print(f"\nStarting training for {total_timesteps:,} steps...")
    print(f"Monitor at: {task.get_output_log_web_page()}\n")
    
    try:
        # Training:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
        
        # Saving final model:
        final_path = f"./models/{config_name}/final_model"
        model.save(final_path)
        task.upload_artifact('final_model', f"{final_path}.zip")
        
        print(f"\n✓ Training completed: {config_name}")
        print(f"  Best model: ./best_models/{config_name}/best_model.zip")
        
        # Getting final evaluation results:
        # Loading evaluation results if available:
        import numpy as np
        eval_file = f"./logs/{config_name}/evaluations.npz"
        if os.path.exists(eval_file):
            eval_data = np.load(eval_file)
            final_reward = eval_data['results'][-1].mean()
            print(f"  Final eval reward: {final_reward:.2f}")
            return final_reward
        
        return None
        
    except KeyboardInterrupt:
        print(f"\n✗ Training interrupted: {config_name}")
        interrupted_path = f"./models/{config_name}/interrupted_model"
        model.save(interrupted_path)
        return None
    
    finally:
        env.close()
        eval_env.close()



# HYPERPARAMETER CONFIGURATIONS TO TEST

# Each configuration is a dictionary of hyperparameters

CONFIGURATIONS = {
    # ==================== BASELINE ====================
    "baseline": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "net_arch": [256, 256],
    },
    
    # ==================== LEARNING RATE VARIANTS ====================
    "high_lr": {
        "learning_rate": 1e-3,  # Higher learning rate
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "net_arch": [256, 256],
    },
    
    "low_lr": {
        "learning_rate": 1e-4,  # Lower learning rate
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "net_arch": [256, 256],
    },
    
    # ==================== NETWORK SIZE VARIANTS ====================
    "small_network": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "net_arch": [128, 128],  # Smaller network
    },
    
    "large_network": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "net_arch": [512, 512],  # Larger network
    },
    
    # ==================== GAMMA VARIANTS ====================
    "high_gamma": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.995,  # More future-focused
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "net_arch": [256, 256],
    },
    
    "low_gamma": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.95,  # More present-focused
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "net_arch": [256, 256],
    },
    
    # ==================== BATCH SIZE VARIANTS ====================
    "small_batch": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 32,  # Smaller batches
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "net_arch": [256, 256],
    },
    
    "large_batch": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 128,  # Larger batches
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "net_arch": [256, 256],
    },
    
    # ==================== COMBINED OPTIMIZATIONS ====================
    "aggressive": {
        "learning_rate": 1e-3,  # High LR
        "n_steps": 4096,  # More steps per update
        "batch_size": 128,  # Large batch
        "gamma": 0.995,  # High gamma
        "gae_lambda": 0.98,
        "clip_range": 0.3,  # Larger clip
        "net_arch": [256, 256],
    },
    
    "conservative": {
        "learning_rate": 1e-4,  # Low LR
        "n_steps": 1024,  # Fewer steps per update
        "batch_size": 32,  # Small batch
        "gamma": 0.95,  # Low gamma
        "gae_lambda": 0.90,
        "clip_range": 0.1,  # Smaller clip
        "net_arch": [128, 128],
    },
}



# GROUP ASSIGNMENT

GROUP_ASSIGNMENTS = {
    "Student_1": ["baseline", "high_lr", "low_lr"],
    "Student_2": ["small_network", "large_network"],
    "Student_3": ["high_gamma", "low_gamma"],
    "Student_4": ["small_batch", "large_batch"],
    "Student_5": ["aggressive", "conservative"],
}


def run_assigned_experiments(student_name, total_timesteps=500_000):
    """
    Running experiments assigned to a specific student
    
    Args:
        student_name: e.g., "Student_1", "Student_2", etc.
        total_timesteps: Training duration (default 500k, use 1M for final)
    """
    if student_name not in GROUP_ASSIGNMENTS:
        print(f"Error: Unknown student name '{student_name}'")
        print(f"Available: {list(GROUP_ASSIGNMENTS.keys())}")
        return
    
    assigned_configs = GROUP_ASSIGNMENTS[student_name]
    
    print("=" * 70)
    print(f"HYPERPARAMETER SEARCH - {student_name}")
    print("=" * 70)
    print(f"Assigned configurations: {len(assigned_configs)}")
    print(f"Training duration: {total_timesteps:,} steps each")
    print(f"Estimated time: {len(assigned_configs) * 2} hours on GPU")
    print("=" * 70)
    
    results = {}
    
    for config_name in assigned_configs:
        config = CONFIGURATIONS[config_name]
        result = train_with_config(config_name, config, total_timesteps)
        results[config_name] = result
    
    # Summary:
    print("\n" + "=" * 70)
    print(f"COMPLETED: {student_name}")
    print("=" * 70)
    for config_name, result in results.items():
        status = f"{result:.2f}" if result else "N/A"
        print(f"  {config_name}: {status}")
    print("=" * 70)
    
    return results


def run_single_experiment(config_name, total_timesteps=500_000):
    """
    Running a single experiment by name
    
    Args:
        config_name: Name of configuration (e.g., "baseline", "high_lr")
        total_timesteps: Training duration
    """
    if config_name not in CONFIGURATIONS:
        print(f"Error: Unknown configuration '{config_name}'")
        print(f"Available: {list(CONFIGURATIONS.keys())}")
        return
    
    config = CONFIGURATIONS[config_name]
    result = train_with_config(config_name, config, total_timesteps)
    return result


# ===========================================================================
# MAIN - HOW TO RUN THIS SCRIPT
# ===========================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter search for RL controller")
    parser.add_argument(
        "--student",
        type=str,
        help="Student name (Student_1, Student_2, etc.)",
        default=None
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Single config name to run (e.g., baseline, high_lr)",
        default=None
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        help="Total timesteps for training",
        default=500_000
    )
    
    args = parser.parse_args()
    
    if args.student:
        # Running all experiments for a student:
        run_assigned_experiments(args.student, args.timesteps)
    
    elif args.config:
        # Running single experiment:
        run_single_experiment(args.config, args.timesteps)
    
    else:
        # No arguments - showing help:
        print("=" * 70)
        print("HYPERPARAMETER SEARCH")
        print("=" * 70)
        print("\nUsage:")
        print("  1. Run experiments for your assigned student:")
        print("     python hyperparameter_search.py --student Student_1")
        print("\n  2. Run a single configuration:")
        print("     python hyperparameter_search.py --config baseline")
        print("\n  3. Specify training duration:")
        print("     python hyperparameter_search.py --student Student_1 --timesteps 1000000")
        print("\nAvailable configurations:")
        for name in CONFIGURATIONS.keys():
            print(f"  - {name}")
        print("\nStudent assignments:")
        for student, configs in GROUP_ASSIGNMENTS.items():
            print(f"  {student}: {configs}")
        print("=" * 70)