"""
Comparing results from hyperparameter search
Generates comparison plots and leaderboard
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


def load_experiment_results():
    """Loading results from all hyperparameter experiments"""
    
    results = {}
    
    # Listing all experiment directories:
    logs_dir = Path("./logs")
    
    for exp_dir in logs_dir.iterdir():
        if exp_dir.is_dir():
            exp_name = exp_dir.name
            eval_file = exp_dir / "evaluations.npz"
            
            if eval_file.exists():
                try:
                    eval_data = np.load(eval_file)
                    
                    # Getting final results:
                    final_rewards = eval_data['results'][-1]
                    mean_reward = final_rewards.mean()
                    std_reward = final_rewards.std()

                    # Getting best reward:
                    all_rewards = eval_data['results']
                    best_reward = np.max([r.mean() for r in all_rewards])
                    
                    results[exp_name] = {
                        'final_mean_reward': mean_reward,
                        'final_std_reward': std_reward,
                        'best_reward': best_reward,
                        'n_evaluations': len(eval_data['results']),
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not load {exp_name}: {e}")
    
    return results


def create_leaderboard(results):
    """Creating leaderboard ranking experiments"""
    
    # Sorting by final mean reward (descending):
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]['final_mean_reward'],
        reverse=True
    )
    
    print("=" * 70)
    print("HYPERPARAMETER SEARCH LEADERBOARD")
    print("=" * 70)
    print(f"{'Rank':<6} {'Configuration':<25} {'Final Reward':<20} {'Best Reward':<15}")
    print("-" * 70)
    
    for rank, (name, data) in enumerate(sorted_results, 1):
        print(f"{rank:<6} {name:<25} "
              f"{data['final_mean_reward']:>8.2f} ± {data['final_std_reward']:>5.2f}   "
              f"{data['best_reward']:>8.2f}")
    
    print("=" * 70)
    
    return sorted_results


def plot_comparison(results):
    """Creating comparison plots"""
    
    names = list(results.keys())
    final_rewards = [results[n]['final_mean_reward'] for n in names]
    std_rewards = [results[n]['final_std_reward'] for n in names]
    best_rewards = [results[n]['best_reward'] for n in names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Final rewards with error bars:
    x = np.arange(len(names))
    ax1.barh(x, final_rewards, xerr=std_rewards, capsize=5, alpha=0.7)
    ax1.set_yticks(x)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('Final Mean Reward')
    ax1.set_title('Final Evaluation Rewards')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.axvline(x=0, color='r', linestyle='--', linewidth=1)
    
    # Best rewards:
    ax2.barh(x, best_rewards, alpha=0.7, color='green')
    ax2.set_yticks(x)
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel('Best Reward During Training')
    ax2.set_title('Peak Performance')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nComparison plot saved to: hyperparameter_comparison.png")


if __name__ == "__main__":
    print("Loading hyperparameter experiment results...")
    
    results = load_experiment_results()
    
    if not results:
        print("No results found! Train some models first.")
    else:
        print(f"Found {len(results)} experiments\n")
        
        # Creating leaderboard:
        sorted_results = create_leaderboard(results)
        
        # Saving to JSON:
        with open('hyperparameter_leaderboard.json', 'w') as f:
            json.dump(dict(sorted_results), f, indent=2)
        
        print("\nLeaderboard saved to: hyperparameter_leaderboard.json")
        
        # Creating plots:
        plot_comparison(results)
        
        # Winner:
        winner_name, winner_data = sorted_results[0]
        print(f"\nWINNER: {winner_name}")
        print(f"   Final reward: {winner_data['final_mean_reward']:.2f} ± {winner_data['final_std_reward']:.2f}")
        print(f"   Best model: ./best_models/{winner_name}/best_model.zip")