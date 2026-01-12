"""
Testing script for trained RL agent
Evaluates the agent on multiple target positions and compares with PID baseline
"""

import numpy as np
import sys
import os
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ot2_gym_wrapper import OT2GymWrapper


def evaluate_rl_agent(model_path, n_episodes=10, render=False, verbose=True):
    """
    Evaluating trained RL agent on multiple episodes
    
    Args:
        model_path: Path to saved model (.zip file)
        n_episodes: Number of episodes to test
        render: Whether to show GUI
        verbose: Print detailed results
        
    Returns:
        dict: Performance metrics
    """
    if verbose:
        print("=" * 70)
        print("EVALUATING RL AGENT")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"Episodes: {n_episodes}")
        print("=" * 70)
    
    # Loading model:
    model = PPO.load(model_path)
    
    # Creating environment:
    render_mode = 'human' if render else None
    env = OT2GymWrapper(
        target_tolerance=0.001,
        max_steps=500,
        render_mode=render_mode
    )
    
    results = {
        'errors': [],
        'steps': [],
        'rewards': [],
        'success': [],
        'trajectories': [],
        'targets': [],
        'final_positions': []
    }
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        episode_reward = 0
        trajectory = []
        
        if verbose:
            print(f"\n--- Episode {episode + 1} ---")
            print(f"Target: {info['target_pos']}")
        
        while not done and step_count < 500:
            # Getting action from trained policy:
            action, _states = model.predict(obs, deterministic=True)
            
            # Executing action:
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            trajectory.append(info['current_pos'].copy())
            step_count += 1
            done = terminated or truncated
        
        # Recording results:
        final_error = info['distance']
        is_success = final_error < 0.001
        
        results['errors'].append(final_error)
        results['steps'].append(step_count)
        results['rewards'].append(episode_reward)
        results['success'].append(is_success)
        results['trajectories'].append(trajectory)
        results['targets'].append(info['target_pos'].copy())
        results['final_positions'].append(info['current_pos'].copy())
        
        if verbose:
            status = "SUCCESS" if is_success else "FAILED"
            print(f"Result: {status}")
            print(f"  Final error: {final_error * 1000:.3f}mm")
            print(f"  Steps taken: {step_count}")
            print(f"  Total reward: {episode_reward:.1f}")
    
    env.close()
    
    # Calculating statistics:
    errors_mm = np.array(results['errors']) * 1000
    success_rate = np.mean(results['success']) * 100
    
    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Episodes: {n_episodes}")
        print(f"Success rate (<1mm): {success_rate:.1f}%")
        print(f"Mean error: {errors_mm.mean():.3f} ± {errors_mm.std():.3f} mm")
        print(f"Median error: {np.median(errors_mm):.3f} mm")
        print(f"Min error: {errors_mm.min():.3f} mm")
        print(f"Max error: {errors_mm.max():.3f} mm")
        print(f"Mean steps: {np.mean(results['steps']):.1f}")
        print(f"Mean reward: {np.mean(results['rewards']):.1f}")
        print("=" * 70)
    
    return results


def plot_trajectory(trajectory, target, title="RL Agent Trajectory"):
    """
    Plotting 3D trajectory of agent movement
    
    Args:
        trajectory: List of [x, y, z] positions
        target: Target position [x, y, z]
    """
    trajectory = np.array(trajectory)
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D plot:
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
             'b-', linewidth=2, label='Trajectory')
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                c='green', s=100, marker='o', label='Start')
    ax1.scatter(target[0], target[1], target[2], 
                c='red', s=100, marker='*', label='Target')
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                c='blue', s=100, marker='x', label='End')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(title)
    ax1.legend()
    
    # Distance over time:
    ax2 = fig.add_subplot(122)
    distances = [np.linalg.norm(pos - target) for pos in trajectory]
    ax2.plot(distances, 'b-', linewidth=2)
    ax2.axhline(y=0.001, color='r', linestyle='--', label='1mm threshold')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Distance to Target (m)')
    ax2.set_title('Distance Over Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('rl_trajectory.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Trajectory plot saved to: rl_trajectory.png")


if __name__ == "__main__":
    # Testing with the trained model:
    model_path = "./best_models/test_training_clearml_v1/best_model.zip"
    
    # Checking if model exists:
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first using test_training_clearml.py")
        sys.exit(1)
    
    # Evaluating:
    results = evaluate_rl_agent(
        model_path=model_path,
        n_episodes=10,
        render=False,  # Setting to True to see visualization
        verbose=True
    )
    
    # Plotting first trajectory:
    print("\nPlotting first trajectory...")
    plot_trajectory(
        results['trajectories'][0],
        results['targets'][0],
        title="RL Agent - Episode 1"
    )
    
    # Saving results:
    import json
    output_file = "rl_evaluation_results.json"
    
    # Converting numpy arrays to lists for JSON serialization:
    json_results = {
        'errors_mm': (np.array(results['errors']) * 1000).tolist(),
        'steps': results['steps'],
        'rewards': results['rewards'],
        'success_rate': np.mean(results['success']) * 100,
        'mean_error_mm': np.mean(results['errors']) * 1000,
        'std_error_mm': np.std(results['errors']) * 1000,
    }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")