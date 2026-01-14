"""
test_wrapper.py
Simple sanity check required by Task 11.

Runs the OT2GymWrapper for 1000 steps with random actions and prints some stats.

Run:
    python test_wrapper.py
"""

import numpy as np
from ot2_gym_wrapper import OT2GymWrapper


def main():
    env = OT2GymWrapper(
        target_tolerance=0.01,
        max_steps=400,
        render_mode=None,
        max_vel=0.4,
    )
    obs, info = env.reset()

    print("START pos:", env.current_pos, "target:", env.target_pos)

    def one_step(a, name):
        pos0 = env.current_pos.copy()
        obs, reward, terminated, truncated, info = env.step(a)
        pos1 = env.current_pos.copy()
        print(f"{name}  action={a[:3]}  delta_pos={(pos1 - pos0)}  new_pos={pos1}")
        return obs, reward, terminated, truncated, info

    # pure axis pushes (no gripper)
    one_step(np.array([ 1, 0, 0, 0], dtype=np.float32), "+X")
    one_step(np.array([ 0, 1, 0, 0], dtype=np.float32), "+Y")
    one_step(np.array([ 0, 0, 1, 0], dtype=np.float32), "+Z")


    distances = []
    successes = 0

    for _ in range(1000):
        pos = env.current_pos.copy()
        tgt = env.target_pos.copy()

        direction = tgt - pos
        norm = np.linalg.norm(direction) + 1e-8
        unit = direction / norm

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[:3] = np.clip(unit, -1.0, 1.0)  # push toward target
        action[3] = 0.0

        obs, reward, terminated, truncated, info = env.step(action)

        dist = float(np.linalg.norm(env.target_pos - env.current_pos))
        distances.append(dist)

        if dist < 0.01:
            successes += 1

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    distances = np.array(distances)
    print("Ran 1000 steps (greedy towards target).")
    print(f"Distance (m): mean={distances.mean():.4f}  min={distances.min():.4f}  max={distances.max():.4f}")
    print(f"Success count (within tolerance): {successes}")



if __name__ == "__main__":
    main()
