"""train_remote_clearml.py

ClearML-ready training script for OT2 PPO controller.

Key fixes vs earlier attempts:
- ClearML remote bootstrap happens BEFORE importing stable-baselines3 (SB3), so
  the remote agent can install pinned requirements first (avoids numpy/pandas ABI issues).
- Avoid nested project names with slashes (can trigger 'project already exists' 400s on some servers).
- Optional requirements file (requirements_clearml.txt) is installed on the agent.
- If ClearML is unreachable (e.g., VPN off), we fall back to local training.

Run locally:
  python train_remote_clearml.py

Queue remotely (requires VPN + clearml-init configured):
  CLEARML_REMOTE=1 CLEARML_QUEUE=default python train_remote_clearml.py
  # or for GPU queue (if available on your dashboard):
  CLEARML_REMOTE=1 CLEARML_QUEUE=gpu python train_remote_clearml.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# -------------------------------
# ClearML bootstrap (MUST be early)
# -------------------------------
def maybe_init_clearml():
    """Init ClearML and optionally enqueue remote execution.

    Returns:
        task (clearml.Task) or None
    """
    if os.environ.get("CLEARML_DISABLE", "0") == "1":
        return None

    try:
        from clearml import Task  # import early

        project_name = os.environ.get("CLEARML_PROJECT", "OT2 RL Controller")
        # Put your name in the task name instead of the project path (avoids server 400 on duplicates)
        default_task_name = "PPO_planar_curriculum_20mm_to_10mm_Kinga"
        task_name = os.environ.get("CLEARML_TASK_NAME", default_task_name)

        task = Task.init(project_name=project_name, task_name=task_name)

        # If you want to run remotely, you MUST be on the university VPN at submission time.
        if os.environ.get("CLEARML_REMOTE", "0") == "1":
            docker_image = os.environ.get("CLEARML_DOCKER", "deanis/2023y2b-rl:latest")
            queue = os.environ.get("CLEARML_QUEUE", "default")

            # Ensure we pin numpy/pandas versions on the agent to avoid ABI mismatches.
            # IMPORTANT: requirements file must be committed to your repo.
            req_file = os.environ.get("CLEARML_REQUIREMENTS", "requirements_clearml.txt")
            if Path(req_file).exists():
                task.set_requirements(requirements_file=req_file)

            task.set_base_docker(docker_image)

            print(f"ClearML: queuing remotely on '{queue}' with docker '{docker_image}'")
            # exit_process=True ensures your LOCAL process stops after enqueuing (so it doesn't train locally)
            task.execute_remotely(queue_name=queue, exit_process=True)

        return task

    except Exception as e:
        print(f"ClearML not available/reachable (continuing locally): {e}")
        return None


# Initialize ClearML (may enqueue and exit)
CLEARML_TASK = maybe_init_clearml()

# -------------------------------
# Heavy imports AFTER ClearML bootstrap
# -------------------------------
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from ot2_gym_wrapper import OT2GymWrapper

# Reduce BLAS thread thrash (helps macOS, harmless elsewhere)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# -------------------------------
# Config
# -------------------------------
SEED = 42
N_ENVS = int(os.environ.get("N_ENVS", "8"))

# Robot control limits (in meters/step on XY; env keeps Z fixed in planar mode)
MAX_VEL = float(os.environ.get("MAX_VEL", "0.4"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "400"))

STAGE1 = dict(tolerance=0.02, timesteps=int(os.environ.get("STAGE1_STEPS", "400000")))
STAGE2 = dict(tolerance=0.01, timesteps=int(os.environ.get("STAGE2_STEPS", "1200000")))

# Where to save outputs
RUN_DIR = Path(os.environ.get("RUN_DIR", "./runs_remote"))
MODELS_DIR = RUN_DIR / "models"
BEST_DIR = RUN_DIR / "best_models"
LOGS_DIR = RUN_DIR / "logs"
VECNORM_PATH = RUN_DIR / "vecnormalize.pkl"

for d in [RUN_DIR, MODELS_DIR, BEST_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def make_env(rank: int, tolerance: float, seed: int = 0):
    """Factory for SubprocVecEnv."""

    def _init():
        env = OT2GymWrapper(
            target_tolerance=tolerance,
            max_steps=MAX_STEPS,
            max_vel=MAX_VEL,
            render_mode=None,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def build_vec_env(tolerance: float):
    set_random_seed(SEED)
    env_fns = [make_env(i, tolerance, SEED) for i in range(N_ENVS)]
    venv = SubprocVecEnv(env_fns, start_method="spawn")
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return venv


def evaluate_greedy(env: OT2GymWrapper, episodes: int = 20):
    """Quick deterministic eval: greedy action towards target (XY)."""
    errs = []
    successes_10mm = 0
    successes_5mm = 0

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        last_dist = None
        while not done:
            # Observation is assumed to include (dx, dy, dz, ...) or similar.
            # We'll use env helper if present; otherwise compute from env state.
            pos = getattr(env, "current_pos", None)
            tgt = getattr(env, "current_target", None)

            if pos is None or tgt is None:
                # Fallback to info dict if wrapper exposes it there
                pos = info.get("current_pos", np.zeros(3))
                tgt = info.get("target_pos", np.zeros(3))

            direction = (tgt - pos).astype(np.float32)
            direction[2] = 0.0  # planar
            norm = float(np.linalg.norm(direction) + 1e-8)
            unit = direction / norm

            action = np.zeros(env.action_space.shape, dtype=np.float32)
            action[:3] = np.clip(unit, -1.0, 1.0)
            if action.shape[0] > 3:
                action[3] = 0.0

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            last_dist = float(info.get("distance", norm))

        if last_dist is None:
            last_dist = float("inf")

        errs.append(last_dist)
        if last_dist < 0.01:
            successes_10mm += 1
        if last_dist < 0.005:
            successes_5mm += 1

    return {
        "mean_error_mm": float(np.mean(errs) * 1000.0),
        "median_error_mm": float(np.median(errs) * 1000.0),
        "success_<10mm": float(successes_10mm / episodes),
        "success_<5mm": float(successes_5mm / episodes),
    }


def main():
    start_time = datetime.now()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("RL TRAINING (PPO) - PLANAR CURRICULUM 20mm -> 10mm")
    print("=" * 70)
    print(f"Device: {device} | N_ENVS={N_ENVS} | MAX_VEL={MAX_VEL} | MAX_STEPS={MAX_STEPS}")
    print(f"Stage1: tol={STAGE1['tolerance']} for {STAGE1['timesteps']:,} steps")
    print(f"Stage2: tol={STAGE2['tolerance']} for {STAGE2['timesteps']:,} steps")
    print("=" * 70)

    train_env = build_vec_env(STAGE1["tolerance"])
    eval_env = build_vec_env(STAGE1["tolerance"])

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(MODELS_DIR),
        name_prefix="ppo_ot2",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(BEST_DIR),
        log_path=str(LOGS_DIR),
        eval_freq=50_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=SEED,
        device=device,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    # ---- Stage 1
    model.learn(
        total_timesteps=STAGE1["timesteps"],
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save VecNormalize stats after stage 1
    train_env.save(str(VECNORM_PATH))

    # ---- Stage 2 (optional)
    if STAGE2["timesteps"] > 0:
        train_env_2 = build_vec_env(STAGE2["tolerance"])
        eval_env_2 = build_vec_env(STAGE2["tolerance"])

        # Transfer running statistics to keep normalization consistent
        train_env_2.obs_rms = train_env.obs_rms
        train_env_2.ret_rms = train_env.ret_rms
        train_env_2.training = True

        eval_env_2.obs_rms = train_env.obs_rms
        eval_env_2.ret_rms = train_env.ret_rms
        eval_env_2.training = False

        model.set_env(train_env_2)

        eval_callback_2 = EvalCallback(
            eval_env_2,
            best_model_save_path=str(BEST_DIR),
            log_path=str(LOGS_DIR),
            eval_freq=50_000,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )

        model.learn(
            total_timesteps=STAGE2["timesteps"],
            callback=[checkpoint_callback, eval_callback_2],
            progress_bar=True,
        )

        # Save updated VecNormalize stats
        train_env_2.save(str(VECNORM_PATH))

    # ---- Save final model
    final_model_path = MODELS_DIR / "final_model.zip"
    model.save(str(final_model_path))

    elapsed = datetime.now() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Time elapsed: {elapsed}")
    print(f"Saved model: {final_model_path}")
    print(f"Saved VecNormalize: {VECNORM_PATH}")

    # Optional: upload artifacts to ClearML (so you can download later from dashboard)
    if CLEARML_TASK is not None:
        try:
            CLEARML_TASK.upload_artifact("final_model", artifact_object=str(final_model_path))
            CLEARML_TASK.upload_artifact("vecnormalize", artifact_object=str(VECNORM_PATH))
        except Exception:
            pass

    # Quick sanity evaluation (greedy baseline)
    try:
        eval_env_single = OT2GymWrapper(
            target_tolerance=0.01, max_steps=MAX_STEPS, max_vel=MAX_VEL, render_mode=None
        )
        metrics = evaluate_greedy(eval_env_single, episodes=20)
        print("\nFINAL GREEDY BASELINE (20 eps)")
        print(metrics)
        eval_env_single.close()
    except Exception as e:
        print(f"Skipping greedy eval due to error: {e}")


if __name__ == "__main__":
    # Required for macOS + SubprocVecEnv
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
