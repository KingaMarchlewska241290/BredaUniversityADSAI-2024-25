"""
Checking where the robot actually spawns
"""
import sys
import os

# Adding parent directory to path:
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from sim_class import Simulation

def get_pipette_position(state):
    return state['robotId_1']['pipette_position']

# Creating simulation:
sim = Simulation(num_agents=1, render=False)

# Stabilizing:
for _ in range(100):
    sim.run([[0.0, 0.0, 0.0, 0]])

# Getting initial position:
state = sim.run([[0.0, 0.0, 0.0, 0]])
pos = get_pipette_position(state)

print(f"Robot spawns at: {pos}")
print(f"  X: {pos[0]:.4f}m")
print(f"  Y: {pos[1]:.4f}m")  
print(f"  Z: {pos[2]:.4f}m")

print(f"\nWorkspace (conservative):")
print(f"  X: (0.05, 0.25)")
print(f"  Y: (0.05, 0.25)")
print(f"  Z: (0.05, 0.18)")

print(f"\nIs starting position inside workspace?")
print(f"  X: {0.05 <= pos[0] <= 0.25}")
print(f"  Y: {0.05 <= pos[1] <= 0.25}")
print(f"  Z: {0.05 <= pos[2] <= 0.18}")

if not (0.05 <= pos[0] <= 0.25 and 0.05 <= pos[1] <= 0.25 and 0.05 <= pos[2] <= 0.18):
    print("\nPROBLEM: Robot starts OUTSIDE the conservative workspace.")
    print("This is why episodes terminate immediately.")
else:
    print("\nRobot starts inside workspace.")