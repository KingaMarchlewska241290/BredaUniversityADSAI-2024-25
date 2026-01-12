"""
Training script for OT-2 RL Controller with ClearML
Uses Stable Baselines 3 with PPO algorithm
Tracks experiments with ClearML
"""

import os
import sys
import numpy as np
from datetime import datetime

# Adding parent directory to path:
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from clearml import Task

from ot2_gym_wrapper import OT2GymWrapper


class ClearMLCallback(BaseCallback):
    """
    Custom callback for logging to ClearML
    """
    def __init__(self, task, verbose=0):
        super().__init__(verbose)
        self.task = task
        self.clearml_logger = task.get_logger()  # Changed from self.logger to self.clearml_logger
        
    def _on_step(self) -> bool:
        # Logging training metrics:
        if len(self.model.ep_info_buffer) > 0:
            # Getting episode statistics:
            ep_info = self.model.ep_info_buffer[-1]

            # Logging to ClearML:
            self.clearml_logger.report_scalar(
                title="Episode Reward",
                series="reward",
                value=ep_info.get('r', 0),
                iteration=self.num_timesteps
            )
            
            self.clearml_logger.report_scalar(
                title="Episode Length",
                series="length",
                value=ep_info.get('l', 0),
                iteration=self.num_timesteps
            )
        
        return True


def make_env():
    """
    Creating and wrapping the environment.
    """
    def _init():
        env = OT2GymWrapper(
            target_tolerance=0.001,  # 1mm
            max_steps=500,
            render_mode=None  # No GUI for training
        )
        env = Monitor(env)
        return env
    return _init


def train_rl_agent(
    # Algorithm hyperparameters:
    algorithm="PPO",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    
    # Network architecture:
    net_arch=[256, 256],
    
    # Training parameters:
    total_timesteps=1_000_000,
    
    # Experiment name:
    experiment_name=None,
    
    # ClearML project name:
    project_name="OT2 RL Controller",
):
    """
    Training an RL agent to control the OT-2 robot.
    
    Args:
        algorithm: RL algorithm to use (currently only PPO supported)
        learning_rate: Learning rate for optimizer
        n_steps: Number of steps to collect before update
        batch_size: Minibatch size for training
        n_epochs: Number of epochs when optimizing
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        net_arch: Network architecture (list of hidden layer sizes)
        total_timesteps: Total training timesteps
        experiment_name: Name for this experiment
        project_name: ClearML project name
    """
    
    # Generating experiment name if not provided:
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"ot2_rl_{algorithm}_{timestamp}"
    
    # Creating directories:
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("best_models", exist_ok=True)
    
    print("=" * 70)
    print("Training OT-2 RL Controller")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Algorithm: {algorithm}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Network architecture: {net_arch}")
    print("=" * 70)
    
    # Initializing ClearML Task:
    task = Task.init(
        project_name=project_name,
        task_name=experiment_name,
        tags=[algorithm, "RL", "OT2"],
    )

    # Logging hyperparameters to ClearML:
    config = {
        "algorithm": algorithm,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "net_arch": net_arch,
        "total_timesteps": total_timesteps,
        "target_tolerance": 0.001,
        "max_steps": 500,
    }
    
    task.connect(config)
    
    print("\nClearML Task initialized.")
    print(f"View experiment at: {task.get_output_log_web_page()}")
    
    print("\nInitializing environments...")
    
    # Creating training environment:
    env = DummyVecEnv([make_env()])

    # Creating evaluation environment:
    eval_env = DummyVecEnv([make_env()])

    print("Environments created successfully.")

    # Creating callbacks:
    print("\nSetting up callbacks...")
    
    # Checkpoint callback - saving model every 10k steps:
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=f"./models/{experiment_name}/",
        name_prefix="ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # Evaluation callback - evaluating and saving best model:
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./best_models/{experiment_name}/",
        log_path=f"./logs/{experiment_name}/",
        eval_freq=5_000,  # Evaluating every 5k steps
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    
    # ClearML callback:
    clearml_callback = ClearMLCallback(task=task, verbose=1)
    
    callbacks = [checkpoint_callback, eval_callback, clearml_callback]
    
    print("Callbacks configured.")
    
    # Initializing the agent:
    print(f"\nInitializing {algorithm} agent...")
    
    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            policy_kwargs=dict(net_arch=net_arch),
            verbose=1,
            tensorboard_log=f"./logs/{experiment_name}/tensorboard/",
        )
    else:
        raise ValueError(f"Algorithm {algorithm} not supported yet!")
    
    print("Agent initialized!")
    print(f"\nPolicy architecture:\n{model.policy}")
    
    # Training the agent:
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    print(f"\nMonitor training progress at: {task.get_output_log_web_page()}")
    print(f"Local logs: ./logs/{experiment_name}/")
    print("\nTraining will save checkpoints every 10,000 steps")
    print("Best model will be saved based on evaluation performance")
    print("=" * 70 + "\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        
        print("\n" + "=" * 70)
        print("Training completed successfully.")
        print("=" * 70)
        
        # Saving final model:
        final_model_path = f"./models/{experiment_name}/final_model"
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
        # Uploading model to ClearML:
        task.upload_artifact(
            name='final_model',
            artifact_object=f"{final_model_path}.zip"
        )
        
        # Getting final evaluation metrics:
        logger = task.get_logger()
        if os.path.exists(f"./logs/{experiment_name}/evaluations.npz"):
            eval_data = np.load(f"./logs/{experiment_name}/evaluations.npz")
            final_mean_reward = eval_data['results'][-1].mean()
            logger.report_single_value(
                name="Final Mean Reward",
                value=final_mean_reward
            )
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving current model...")
        interrupted_model_path = f"./models/{experiment_name}/interrupted_model"
        model.save(interrupted_model_path)
        print(f"Model saved to: {interrupted_model_path}")
        
        task.upload_artifact(
            name='interrupted_model',
            artifact_object=f"{interrupted_model_path}.zip"
        )
    
    finally:
        # Cleaning up:
        env.close()
        eval_env.close()
        
        print("\nTraining session ended.")
        print(f"Models saved in: ./models/{experiment_name}/")
        print(f"Best model saved in: ./best_models/{experiment_name}/")
        print(f"Logs saved in: ./logs/{experiment_name}/")
        print(f"\nView results at: {task.get_output_log_web_page()}")


if __name__ == "__main__":
    # Example: Train with default parameters:
    train_rl_agent(
        algorithm="PPO",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        net_arch=[256, 256],
        total_timesteps=1_000_000,  # 1M steps
        experiment_name="ppo_baseline_v1",
        project_name="OT2 RL Controller"
    )