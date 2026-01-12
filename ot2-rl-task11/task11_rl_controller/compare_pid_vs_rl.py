"""
Comparing RL controller vs PID controller performance
Side-by-side evaluation on the same target positions
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from pathlib import Path

# Adding paths:
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from ot2_gym_wrapper import OT2GymWrapper
from sim_class import Simulation


class PID:
    """Simple PID controller (from Task 10)"""
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.previous_error = 0.0
        self.integral_limit = 1.0
        
    def compute(self, error, dt=1.0):
        P = self.Kp * error
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        I = self.Ki * self.integral
        derivative = (error - self.previous_error) / dt
        D = self.Kd * derivative
        self.previous_error = error
        return P + I + D
    
    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0


class PIDController3D:
    """3-axis PID controller"""
    def __init__(self, Kp, Ki, Kd):
        self.pid_x = PID(Kp, Ki, Kd)
        self.pid_y = PID(Kp, Ki, Kd)
        self.pid_z = PID(Kp, Ki, Kd)
        
    def compute(self, target_pos, current_pos):
        errors = [target_pos[i] - current_pos[i] for i in range(3)]
        velocities = [
            self.pid_x.compute(errors[0]),
            self.pid_y.compute(errors[1]),
            self.pid_z.compute(errors[2])
        ]
        return velocities
    
    def reset(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()


def evaluate_pid(targets, max_steps=500, verbose=True):
    """Evaluating PID controller on target positions"""
    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATING PID CONTROLLER")
        print("=" * 70)
    
    sim = Simulation(num_agents=1, render=False)
    
    # Stabilizing:
    for _ in range(100):
        sim.run([[0.0, 0.0, 0.0, 0]])
    
    # Creating controller with my optimized gains:
    controller = PIDController3D(Kp=4.0, Ki=0.5, Kd=0.2)
    
    results = {
        'errors': [],
        'steps': [],
        'success': [],
        'trajectories': []
    }
    
    for i, target in enumerate(targets):
        if verbose:
            print(f"\nTarget {i+1}: {target}")
        
        controller.reset()
        trajectory = []
        
        for step in range(max_steps):
            state = sim.run([[0.0, 0.0, 0.0, 0]])
            current_pos = state['robotId_1']['pipette_position']
            trajectory.append(current_pos.copy())
            
            distance = np.linalg.norm(np.array(current_pos) - np.array(target))
            
            if distance < 0.001:
                if verbose:
                    print(f"Reached in {step} steps, error: {distance*1000:.3f}mm")
                
                results['errors'].append(distance)
                results['steps'].append(step)
                results['success'].append(True)
                results['trajectories'].append(trajectory)
                break
            
            # PID control:
            velocities = controller.compute(target, current_pos)
            max_vel = 0.7
            velocities = [max(-max_vel, min(max_vel, v)) for v in velocities]
            
            sim.run([velocities + [0]])
        
        else:
            # Timeout:
            state = sim.run([[0.0, 0.0, 0.0, 0]])
            final_pos = state['robotId_1']['pipette_position']
            distance = np.linalg.norm(np.array(final_pos) - np.array(target))
            
            if verbose:
                print(f"  ✗ Timeout, error: {distance*1000:.3f}mm")
            
            results['errors'].append(distance)
            results['steps'].append(max_steps)
            results['success'].append(False)
            results['trajectories'].append(trajectory)
    
    return results


def evaluate_rl(model_path, targets, verbose=True):
    """Evaluating RL agent on target positions"""
    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATING RL CONTROLLER")
        print("=" * 70)
    
    model = PPO.load(model_path)
    env = OT2GymWrapper(target_tolerance=0.001, max_steps=500, render_mode=None)
    
    results = {
        'errors': [],
        'steps': [],
        'success': [],
        'trajectories': []
    }
    
    for i, target in enumerate(targets):
        if verbose:
            print(f"\nTarget {i+1}: {target}")
        
        # Setting specific target (need to modify reset for this):
        obs, info = env.reset()
        env.target_pos = np.array(target)  # Override target
        obs = env._get_observation()
        
        trajectory = []
        done = False
        step_count = 0
        
        while not done and step_count < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            trajectory.append(info['current_pos'].copy())
            step_count += 1
            done = terminated or truncated
        
        final_error = info['distance']
        is_success = final_error < 0.001
        
        if verbose:
            status = "+" if is_success else "-"
            print(f"  {status} Steps: {step_count}, error: {final_error*1000:.3f}mm")
        
        results['errors'].append(final_error)
        results['steps'].append(step_count)
        results['success'].append(is_success)
        results['trajectories'].append(trajectory)
    
    env.close()
    return results


def plot_comparison(pid_results, rl_results):
    """Creating comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Error comparison:
    ax = axes[0, 0]
    errors_pid = np.array(pid_results['errors']) * 1000
    errors_rl = np.array(rl_results['errors']) * 1000
    
    x = np.arange(len(errors_pid))
    width = 0.35
    
    ax.bar(x - width/2, errors_pid, width, label='PID', alpha=0.8)
    ax.bar(x + width/2, errors_rl, width, label='RL', alpha=0.8)
    ax.axhline(y=1.0, color='r', linestyle='--', label='1mm threshold')
    ax.set_xlabel('Target Position')
    ax.set_ylabel('Final Error (mm)')
    ax.set_title('Positioning Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Steps comparison:
    ax = axes[0, 1]
    ax.bar(x - width/2, pid_results['steps'], width, label='PID', alpha=0.8)
    ax.bar(x + width/2, rl_results['steps'], width, label='RL', alpha=0.8)
    ax.set_xlabel('Target Position')
    ax.set_ylabel('Steps to Reach Target')
    ax.set_title('Convergence Speed Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Success rate:
    ax = axes[1, 0]
    success_rates = [
        np.mean(pid_results['success']) * 100,
        np.mean(rl_results['success']) * 100
    ]
    ax.bar(['PID', 'RL'], success_rates, alpha=0.8, color=['blue', 'orange'])
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate (<1mm precision)')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(success_rates):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Mean error box plot:
    ax = axes[1, 1]
    ax.boxplot([errors_pid, errors_rl], labels=['PID', 'RL'])
    ax.axhline(y=1.0, color='r', linestyle='--', label='1mm threshold')
    ax.set_ylabel('Final Error (mm)')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('pid_vs_rl_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nComparison plot saved to: pid_vs_rl_comparison.png")


if __name__ == "__main__":
    # Defining test targets (same positions for fair comparison):
    test_targets = [
        [0.15, 0.12, 0.15],
        [0.10, 0.10, 0.16],
        [0.20, 0.15, 0.14],
        [0.12, 0.18, 0.17],
        [0.18, 0.10, 0.15],
    ]
    
    # Checking if RL model exists:
    rl_model_path = "./best_models/test_training_clearml_v1/best_model.zip"
    if not Path(rl_model_path).exists():
        print(f"Error: RL model not found at {rl_model_path}")
        print("Please train a model first using test_training_clearml.py")
        sys.exit(1)
    
    print("=" * 70)
    print("PID vs RL CONTROLLER COMPARISON")
    print("=" * 70)
    print(f"Testing on {len(test_targets)} positions")
    print(f"Target tolerance: 1mm")
    print(f"Max steps: 500")
    
    # Evaluating both controllers:
    pid_results = evaluate_pid(test_targets, verbose=True)
    rl_results = evaluate_rl(rl_model_path, test_targets, verbose=True)
    
    # Printing summary:
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    pid_errors = np.array(pid_results['errors']) * 1000
    rl_errors = np.array(rl_results['errors']) * 1000
    
    print(f"\nPID Controller:")
    print(f"  Success rate: {np.mean(pid_results['success'])*100:.1f}%")
    print(f"  Mean error: {pid_errors.mean():.3f} ± {pid_errors.std():.3f} mm")
    print(f"  Mean steps: {np.mean(pid_results['steps']):.1f}")
    
    print(f"\nRL Controller:")
    print(f"  Success rate: {np.mean(rl_results['success'])*100:.1f}%")
    print(f"  Mean error: {rl_errors.mean():.3f} ± {rl_errors.std():.3f} mm")
    print(f"  Mean steps: {np.mean(rl_results['steps']):.1f}")
    
    print("\n" + "=" * 70)
    
    # Creating comparison plots:
    plot_comparison(pid_results, rl_results)
    
    # Saving results:
    import json
    results_summary = {
        'pid': {
            'success_rate': float(np.mean(pid_results['success']) * 100),
            'mean_error_mm': float(pid_errors.mean()),
            'std_error_mm': float(pid_errors.std()),
            'mean_steps': float(np.mean(pid_results['steps']))
        },
        'rl': {
            'success_rate': float(np.mean(rl_results['success']) * 100),
            'mean_error_mm': float(rl_errors.mean()),
            'std_error_mm': float(rl_errors.std()),
            'mean_steps': float(np.mean(rl_results['steps']))
        }
    }
    
    with open('pid_vs_rl_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\nResults saved to: pid_vs_rl_results.json")