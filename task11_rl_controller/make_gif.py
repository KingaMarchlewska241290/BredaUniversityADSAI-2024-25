"""
make_gif.py
Creates a GIF of the trained agent reaching targets (Task 11 deliverable).

This script depends on your simulator supporting rendering. If OT2Simulation can render
frames, adapt `get_frame()` below accordingly.

If rendering isn't supported on your machine, run this on the GPU server where PyBullet GUI works.

Run:
    python make_gif.py

Output:
    rl_agent.gif
"""

import numpy as np
import imageio.v2 as imageio
from stable_baselines3 import PPO

from ot2_gym_wrapper import OT2GymWrapper


def main():
    # IMPORTANT: set render_mode="human" if your sim opens a GUI,
    # or adapt the sim to return frames.
    env = OT2GymWrapper(target_tolerance=0.005, max_steps=300, render_mode="human")

    model = PPO.load("./models_local/final_model_5mm.zip")

    frames = []
    obs, info = env.reset()

    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # ---- Frame capture (adapt if needed) ----
        # If your sim exposes a method like sim.get_camera_image(), use it here.
        try:
            frame = env.sim.get_camera_image()  # type: ignore
            frames.append(frame)
        except Exception:
            pass

        if terminated or truncated:
            break

    env.close()

    if not frames:
        print("No frames captured. If your sim doesn't provide frames, record your screen instead.")
        return

    imageio.mimsave("rl_agent.gif", frames, fps=20)
    print("Saved: rl_agent.gif")


if __name__ == "__main__":
    main()
