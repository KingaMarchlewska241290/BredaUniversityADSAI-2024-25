"""
Test script for OT-2 Gymnasium Wrapper
Runs the environment for 1000 steps with random actions
"""

import numpy as np
from ot2_gym_wrapper import OT2GymWrapper


def test_random_actions(total_steps=1000):
    """
    Testing the wrapper with random actions.
    
    Args:
        total_steps: Number of steps to run
    """
    print("=" * 60)
    print("Testing OT-2 Gymnasium Wrapper")
    print("=" * 60)
    
    # Initializing environment:
    env = OT2GymWrapper()
    
    print(f"\nObservation space: {env.observation_space}")
    print(f"Observation shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Action shape: {env.action_space.shape}")
    
    # Reseting environment:
    obs, info = env.reset()
    print(f"\nInitial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Running random actions:
    steps_completed = 0
    episodes_completed = 0
    episode_rewards = []
    episode_steps = []
    current_episode_reward = 0
    current_episode_steps = 0
    
    print(f"\nRunning {total_steps} steps with random actions...")
    print("-" * 60)
    
    while steps_completed < total_steps:
        # Sampling random action:
        action = env.action_space.sample()
        
        # Executing action:
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_episode_reward += reward
        current_episode_steps += 1
        steps_completed += 1
        
        # Printing progress every 100 steps:
        if steps_completed % 100 == 0:
            print(f"Step {steps_completed}/{total_steps} - "
                  f"Distance: {info['distance']:.6f}m - "
                  f"Episode: {episodes_completed + 1}")
        
        # Checking if episode ended:
        if terminated or truncated:
            episodes_completed += 1
            episode_rewards.append(current_episode_reward)
            episode_steps.append(current_episode_steps)
            
            termination_reason = "Success" if terminated else "Truncated"
            print(f"\n  Episode {episodes_completed} ended ({termination_reason})")
            print(f"    Steps: {current_episode_steps}")
            print(f"    Total reward: {current_episode_reward:.2f}")
            print(f"    Final distance: {info['distance']:.6f}m")
            print(f"    Success: {info['is_success']}")
            print("-" * 60)
            
            # Resetting episode counters:
            current_episode_reward = 0
            current_episode_steps = 0

            # Resetting environment for next episode:
            obs, info = env.reset()
    
    # Printing summary:
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Total steps completed: {steps_completed}")
    print(f"Total episodes: {episodes_completed}")
    
    if episode_rewards:
        print(f"\nEpisode Statistics:")
        print(f"  Mean reward: {np.mean(episode_rewards):.2f}")
        print(f"  Std reward: {np.std(episode_rewards):.2f}")
        print(f"  Mean steps per episode: {np.mean(episode_steps):.1f}")
        print(f"  Std steps per episode: {np.std(episode_steps):.1f}")
    
    env.close()
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_random_actions(total_steps=1000)