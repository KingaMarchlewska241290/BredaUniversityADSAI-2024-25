"""
train_remote.py (also runs locally)
PPO training for OT-2 RL controller with ClearML remote execution support.

How to run locally (no remote queue):
    python train_remote.py

How to enqueue remotely (university ClearML):
    CLEARML_REMOTE=1 python train_remote.py
Optional:
    CLEARML_QUEUE=default|gpu
    CLEARML_DOCKER=deanis/2023y2b-rl:latest

Outputs (created in the run working directory):
- ./models_local/final_model.zip
- ./models_local/vecnormalize.pkl
- ./best_models_local/best_model.zip
- ./logs_local/ (monitor + eval logs)
- ./training_results.json

Notes:
- ClearML remote runs will use the latest COMMITTED git version of this script + ot2_gym_wrapper.py.
  Make sure your planar-fixed wrapper is committed before queuing.
"""

from __future__ import annotations

import os
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Reduce BLAS thread thrash (helps macOS, harmless elsewhere)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv

# Your wrapper must be in the repo next to this file as: ot2_gym_wrapper.py
from ot2_gym_wrapper import OT2GymWrapper


# -------------------------
# ClearML (optional local logging + remote queue)
# -------------------------
def init_clearml(project: str, task_name: str, tags: list[str]):
    """
    University-required remote setup:
      task.set_base_docker('deanis/2023y2b-rl:latest')
      task.execute_remotely(queue_name='default')

    Remote is enabled by env var: CLEARML_REMOTE=1
    """
    try:
        from clearml import Task  # type: ignore


        # Normalize project path; ClearML uses "/" as folder separator.
        # Your previous value "OT2 RL Controller / Kinga" can conflict with existing "OT2 RL Controller/Kinga".
        project_norm = project.replace(" / ", "/").replace("/ ", "/").replace(" /", "/")
        project = project_norm

        try:
            task = Task.init(
                project_name=project,
                task_name=task_name,
                tags=tags,
                reuse_last_task_id=False,  # avoid "overwriting/reusing"
            )
        except Exception as e_init:
            # Some ClearML servers return 400 when trying to "create" a project that already exists.
            # If that happens, fall back to a child project folder so the task can still be created.
            if "Project with the same name already exists" in str(e_init):
                project_fallback = f"{project}/Experiments"
                task = Task.init(
                    project_name=project_fallback,
                    task_name=task_name,
                    tags=tags,
                    reuse_last_task_id=False,
                )
            else:
                raise

        docker_image = os.environ.get("CLEARML_DOCKER", "deanis/2023y2b-rl:latest")
        queue = os.environ.get("CLEARML_QUEUE", "default")

        # set docker image (works for remote agents; safe locally too)
        try:
            task.set_base_docker(docker_image)
        except Exception:
            pass

        if os.environ.get("CLEARML_REMOTE", "0") == "1":
            print(f"ClearML: queuing remotely on '{queue}' with docker '{docker_image}'")
            task.execute_remotely(queue_name=queue, exit_process=True)

        return task
    except Exception as e:
        print(f"ClearML not available/reachable (continuing locally): {e}")
        return None


# -------------------------
# Env helpers
# -------------------------
def make_env(target_tolerance: float, max_steps: int, seed: int, max_vel: float):
    def _init():
        env = OT2GymWrapper(
            target_tolerance=target_tolerance,
            max_steps=max_steps,
            render_mode=None,
            seed=seed,
            max_vel=max_vel,
        )
        return Monitor(env)
    return _init


def build_vec_env(
    tolerance: float,
    n_envs: int,
    max_steps: int,
    seed: int,
    max_vel: float,
    use_subproc: bool = True,
):
    env_fns = [make_env(tolerance, max_steps, seed + i, max_vel) for i in range(n_envs)]

    # Subproc is much faster on multi-core CPUs (remote). On some mac setups, DummyVecEnv can be stabler.
    if use_subproc:
        start_method = "spawn" if platform.system().lower() == "darwin" else "fork"
        venv = SubprocVecEnv(env_fns, start_method=start_method)
    else:
        venv = DummyVecEnv(env_fns)

    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return venv


def evaluate(model: PPO, tol: float, episodes: int, max_steps: int) -> dict:
    env = OT2GymWrapper(target_tolerance=tol, max_steps=max_steps, render_mode=None)
    errors_mm = []
    success = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        err_mm = float(info["distance"]) * 1000.0
        errors_mm.append(err_mm)
        success.append(err_mm < (tol * 1000.0))
    env.close()

    arr = np.array(errors_mm, dtype=float)
    return {
        "tol_m": float(tol),
        "episodes": int(episodes),
        "mean_error_mm": float(arr.mean()),
        "median_error_mm": float(np.median(arr)),
        "min_error_mm": float(arr.min()),
        "max_error_mm": float(arr.max()),
        "success_rate_pct": float(np.mean(success) * 100.0),
        "all_errors_mm": [float(x) for x in arr.tolist()],
    }


def points_from_mean_error_mm(mean_error_mm: float) -> int:
    mean_error_m = mean_error_mm / 1000.0
    if mean_error_m < 0.001:
        return 8
    if mean_error_m < 0.005:
        return 6
    if mean_error_m < 0.01:
        return 4
    return 0


def main() -> None:
    # -------------------------
    # Config (matches the specs you were training with locally)
    # -------------------------
    SEED = 42
    set_random_seed(SEED)

    N_ENVS = 8
    MAX_VEL = 0.4      # (your wrapper uses max_vel; it's the effective delta/step scale)
    MAX_STEPS = 400

    # Curriculum tuned for "get <10mm first"
    STAGE1 = dict(tolerance=0.02, timesteps=400_000)   # easier
    STAGE2 = dict(tolerance=0.01, timesteps=1_200_000) # score threshold

    PPO_KWARGS = dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        device="cpu",  # PPO+MLP is usually faster on CPU
    )

    # -------------------------
    # Output folders
    # -------------------------
    Path("models_local").mkdir(exist_ok=True)
    Path("best_models_local").mkdir(exist_ok=True)
    Path("logs_local").mkdir(exist_ok=True)

    # -------------------------
    # ClearML (local logging or remote queue)
    # -------------------------
    task = init_clearml(
        project="OT2 RL Controller/Kinga",
        task_name="PPO_planar_curriculum_20mm_to_10mm",
        tags=["ppo", "planar", "cpu", "curriculum"],
    )
    if task is not None:
        task.connect(
            {
                "PPO": PPO_KWARGS,
                "STAGE1": STAGE1,
                "STAGE2": STAGE2,
                "N_ENVS": N_ENVS,
                "MAX_VEL": MAX_VEL,
                "MAX_STEPS": MAX_STEPS,
            }
        )

    print("=" * 70)
    print("RL TRAINING (PPO) - PLANAR CURRICULUM 20mm -> 10mm")
    print("=" * 70)
    print(f"Device: cpu | N_ENVS={N_ENVS} | MAX_VEL={MAX_VEL} | MAX_STEPS={MAX_STEPS}")
    print(f"Stage1: tol={STAGE1['tolerance']} for {STAGE1['timesteps']:,} steps")
    print(f"Stage2: tol={STAGE2['tolerance']} for {STAGE2['timesteps']:,} steps")
    if task is not None:
        try:
            print(f"ClearML: {task.get_output_log_web_page()}")
        except Exception:
            pass
    print("=" * 70)

    # -------------------------
    # Vec envs + normalize
    # -------------------------
    use_subproc = True
    train_env = build_vec_env(STAGE1["tolerance"], N_ENVS, MAX_STEPS, SEED, MAX_VEL, use_subproc=use_subproc)
    eval_env = build_vec_env(STAGE1["tolerance"], N_ENVS, MAX_STEPS, SEED + 10_000, MAX_VEL, use_subproc=use_subproc)

    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path="./models_local/", name_prefix="ckpt")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_models_local/",
        log_path="./logs_local/",
        eval_freq=100_000,      # cheaper than 50k
        n_eval_episodes=3,      # cheaper than 5
        deterministic=True,
        render=False,
    )

    model = PPO("MlpPolicy", train_env, **PPO_KWARGS)

    start_time = datetime.now()

    # Stage 1
    model.learn(total_timesteps=STAGE1["timesteps"], callback=[checkpoint_callback, eval_callback], progress_bar=True)

    # Stage 2: swap tolerance, keep VecNormalize stats
    train_env_2 = build_vec_env(STAGE2["tolerance"], N_ENVS, MAX_STEPS, SEED, MAX_VEL, use_subproc=use_subproc)
    eval_env_2 = build_vec_env(STAGE2["tolerance"], N_ENVS, MAX_STEPS, SEED + 10_000, MAX_VEL, use_subproc=use_subproc)

    train_env_2.obs_rms = train_env.obs_rms
    train_env_2.ret_rms = train_env.ret_rms
    train_env_2.training = True

    eval_env_2.obs_rms = train_env.obs_rms
    eval_env_2.ret_rms = train_env.ret_rms
    eval_env_2.training = False

    model.set_env(train_env_2)

    eval_callback_2 = EvalCallback(
        eval_env_2,
        best_model_save_path="./best_models_local/",
        log_path="./logs_local/",
        eval_freq=100_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=STAGE2["timesteps"], callback=[checkpoint_callback, eval_callback_2], progress_bar=True)

    elapsed = datetime.now() - start_time

    # Save model + VecNormalize
    model_path = Path("models_local") / "final_model"
    vecnorm_path = Path("models_local") / "vecnormalize.pkl"
    model.save(str(model_path))
    train_env_2.save(str(vecnorm_path))

    # Upload artifacts to ClearML (so you can download later from the UI)
    if task is not None:
        try:
            task.upload_artifact("final_model", artifact_object=str(model_path) + ".zip")
            task.upload_artifact("vecnormalize", artifact_object=str(vecnorm_path))
        except Exception:
            pass

    # Final evaluation: score at 10mm + optional at 5mm
    eval_10 = evaluate(model, tol=0.01, episodes=20, max_steps=MAX_STEPS)
    eval_5 = evaluate(model, tol=0.005, episodes=20, max_steps=MAX_STEPS)

    results = {
        "training_time": str(elapsed),
        "eval_10mm": eval_10,
        "eval_5mm": eval_5,
        "points_from_mean_error_10mm": points_from_mean_error_mm(eval_10["mean_error_mm"]),
        "config": {
            "PPO": PPO_KWARGS,
            "STAGE1": STAGE1,
            "STAGE2": STAGE2,
            "N_ENVS": N_ENVS,
            "MAX_VEL": MAX_VEL,
            "MAX_STEPS": MAX_STEPS,
        },
        "saved": {
            "model_zip": str(model_path) + ".zip",
            "vecnormalize": str(vecnorm_path),
            "best_model_zip": "./best_models_local/best_model.zip",
        },
    }

    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Time elapsed: {elapsed}")
    print(f"Saved model: {model_path}.zip")
    print(f"Saved VecNormalize: {vecnorm_path}")
    print("Saved results: training_results.json")

    print("\n" + "=" * 70)
    print("FINAL EVAL (10mm)")
    print("=" * 70)
    print(f"Mean error:   {eval_10['mean_error_mm']:.2f} mm")
    print(f"Median error: {eval_10['median_error_mm']:.2f} mm")
    print(f"Success <10mm: {eval_10['success_rate_pct']:.0f}%")
    print(f"Points (mean-error rubric): {results['points_from_mean_error_10mm']}/8")

    print("\n" + "=" * 70)
    print("FINAL EVAL (5mm) [optional]")
    print("=" * 70)
    print(f"Mean error:   {eval_5['mean_error_mm']:.2f} mm")
    print(f"Median error: {eval_5['median_error_mm']:.2f} mm")
    print(f"Success <5mm: {eval_5['success_rate_pct']:.0f}%")

    # Cleanup
    try:
        train_env.close(); eval_env.close(); train_env_2.close(); eval_env_2.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()