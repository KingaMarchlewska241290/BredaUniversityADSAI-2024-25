"""
plot_results.py
Generates simple plots required by Task 11:
- Learning curve (from SB3 monitor logs if available)
- Error distribution from local_training_results.json or test_results.json

Run:
    python plot_results.py

Outputs:
    plots/learning_curve.png
    plots/error_hist.png
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.results_plotter import load_results, ts2xy


def plot_learning_curve(log_dir: str, out_path: Path) -> None:
    try:
        x, y = ts2xy(load_results(log_dir), "timesteps")
        plt.figure()
        plt.plot(x, y)
        plt.xlabel("Timesteps")
        plt.ylabel("Episode reward")
        plt.title("Training reward (Monitor)")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"Could not plot learning curve from {log_dir}: {e}")


def plot_error_hist(results_path: Path, out_path: Path) -> None:
    with open(results_path, "r") as f:
        data = json.load(f)
    errors = np.array(data.get("all_errors_mm", []), dtype=float)
    if errors.size == 0:
        print(f"No errors found in {results_path}")
        return
    plt.figure()
    plt.hist(errors, bins=15)
    plt.xlabel("Final error (mm)")
    plt.ylabel("Count")
    plt.title(f"Error distribution (n={len(errors)})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    Path("plots").mkdir(exist_ok=True)

    plot_learning_curve("logs_local", Path("plots/learning_curve.png"))

    if Path("local_training_results.json").exists():
        plot_error_hist(Path("local_training_results.json"), Path("plots/error_hist.png"))
        print("Saved: plots/error_hist.png")
    elif Path("test_results.json").exists():
        plot_error_hist(Path("test_results.json"), Path("plots/error_hist.png"))
        print("Saved: plots/error_hist.png")


if __name__ == "__main__":
    main()
