"""
test_trained_model.py
Required by Task 11: demonstrates how to load and evaluate a trained model.

This expects:
- models_local/final_model_5mm.zip
Optionally:
- models_local/vecnormalize.pkl  (if you trained with VecNormalize)

Run:
    python test_trained_model.py
"""

import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from ot2_gym_wrapper import OT2GymWrapper


def make_env(tol=0.005, max_steps=300):
    def _init():
        env = OT2GymWrapper(target_tolerance=tol, max_steps=max_steps, render_mode=None)
        return Monitor(env)
    return _init


def main():
    tol = 0.005
    max_steps = 300
    episodes = 30

    env = DummyVecEnv([make_env(tol, max_steps)])

    # Load normalization stats if present
    try:
        env = VecNormalize.load("./models_local/vecnormalize.pkl", env)
        env.training = False
        env.norm_reward = False
        print("Loaded VecNormalize stats.")
    except Exception:
        print("No VecNormalize stats found; evaluating without normalization.")

    model = PPO.load("./models_local/final_model_5mm.zip", env=env)

    errors_mm = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            info = infos[0]
        err_mm = float(info["distance"]) * 1000.0
        errors_mm.append(err_mm)
        print(f"Ep {ep+1:2d}: {err_mm:7.2f} mm")

    errors_mm = np.array(errors_mm)
    results = {
        "episodes": episodes,
        "mean_error_mm": float(errors_mm.mean()),
        "median_error_mm": float(np.median(errors_mm)),
        "min_error_mm": float(errors_mm.min()),
        "max_error_mm": float(errors_mm.max()),
        "success_rate_5mm": float(np.mean(errors_mm < 5.0) * 100.0),
        "success_rate_10mm": float(np.mean(errors_mm < 10.0) * 100.0),
        "all_errors_mm": [float(x) for x in errors_mm.tolist()],
    }
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved: test_results.json")
    print(f"Mean error: {results["mean_error_mm"]:.2f} mm")


if __name__ == "__main__":
    main()
