"""
Debug script to see what's happening in the environment
"""
from ot2_gym_wrapper import OT2GymWrapper
import numpy as np

env = OT2GymWrapper(render_mode=None)

print("Testing wrapper with a few steps...")
obs, info = env.reset()

print(f"\nInitial state:")
print(f"  Current pos: {info['current_pos']}")
print(f"  Target pos: {info['target_pos']}")
print(f"  Distance: {info['distance']:.4f}m")
print(f"  Workspace limits: {env.workspace_limits}")

for step in range(5):
    # Taking a random action:
    action = env.action_space.sample()
    print(f"\nStep {step + 1}:")
    print(f"  Action (before clip/scale): {action}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"  Current pos: {info['current_pos']}")
    print(f"  Out of bounds: {env._is_out_of_bounds(info['current_pos'])}")
    print(f"  Distance: {info['distance']:.4f}m")
    print(f"  Reward: {reward:.2f}")
    print(f"  Terminated: {terminated}, Truncated: {truncated}")
    
    if terminated or truncated:
        print(f"  Episode ended.")
        break

env.close()