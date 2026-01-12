"""
Quick test of training setup with ClearML
Runs for only 10,000 steps to verify everything works
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_rl_agent_clearml import train_rl_agent

if __name__ == "__main__":
    print("Testing training setup with ClearML...")
    print("This will run for only 10,000 steps.")
    
    train_rl_agent(
        algorithm="PPO",
        learning_rate=3e-4,
        n_steps=512,  # Smaller for faster testing
        batch_size=64,
        total_timesteps=10_000,  # Short test
        experiment_name="test_training_clearml_v1",
        project_name="OT2 RL Controller - Tests"
    )