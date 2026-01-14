"""
train_local_clearml.py

Single entrypoint that:
- Trains LOCALLY by default (no ClearML required)
- If CLEARML_REMOTE=1, queues the same training job to ClearML Agent
- Works around the common remote docker NumPy/Pandas ABI mismatch by force-reinstalling
  pinned wheels from requirements_clearml.txt *before* importing stable-baselines3.

Usage:
  # local:
  python train_local_clearml.py

  # remote (queue_name can be "default" or "gpu" if available):
  CLEARML_REMOTE=1 CLEARML_QUEUE=default python train_local_clearml.py
"""

from __future__ import annotations

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# ----------------------------
# 0) Optional: ClearML remote
# ----------------------------
CLEARML_REMOTE = os.getenv("CLEARML_REMOTE", "0") == "1"
CLEARML_QUEUE = os.getenv("CLEARML_QUEUE", "default")
PROJECT_NAME = os.getenv("CLEARML_PROJECT", "OT2 RL Controller")  # avoid "/" in project name
TASK_NAME = os.getenv("CLEARML_TASK_NAME", "PPO_planar_curriculum_20mm_to_10mm")
BASE_DOCKER = os.getenv("CLEARML_DOCKER", "deanis/2023y2b-rl:latest")

def _maybe_bootstrap_requirements() -> None:
    """
    In the remote docker, the base image sometimes has NumPy/Pandas compiled against
    different ABIs -> 'numpy.dtype size changed' errors.
    This installs pinned wheels BEFORE importing SB3 (which imports pandas).
    """
    req_file = Path(__file__).with_name("requirements_clearml.txt")
    if not req_file.exists():
        # No requirements file; nothing to bootstrap.
        return

    # Heuristic: only do the aggressive reinstall when running under ClearML agent.
    # ClearML sets at least one of these in most setups.
    running_under_agent = (
        os.getenv("CLEARML_WORKER_ID") is not None
        or os.getenv("CLEARML_AGENT_EXECUTION") is not None
        or os.getenv("CLEARML_DOCKER_IMAGE") is not None
    )
    if not running_under_agent:
        return

    print(f"[bootstrap] Detected ClearML agent environment. Installing pinned requirements from: {req_file}")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--upgrade",
        "--force-reinstall",
        "--no-cache-dir",
        "-r", str(req_file),
    ]
    subprocess.check_call(cmd)


def _maybe_init_clearml_and_queue() -> None:
    """
    If CLEARML_REMOTE=1, initialize ClearML and queue the script remotely.
    If ClearML is unavailable (e.g., not on VPN), it will fall back to local training.
    """
    if not CLEARML_REMOTE:
        return

    try:
        from clearml import Task  # noqa: F401
    except Exception as e:
        print(f"ClearML not installed/available; continuing locally. ({e})")
        return

    try:
        from clearml import Task

        task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)

        # Set docker image for remote agent
        task.set_base_docker(BASE_DOCKER)

        # IMPORTANT: execute_remotely will terminate the local process after enqueueing.
        print(f"[ClearML] Queueing remotely on '{CLEARML_QUEUE}' with docker '{BASE_DOCKER}' ...")
        task.execute_remotely(queue_name=CLEARML_QUEUE)

    except Exception as e:
        # If you're not connected to VPN / ClearML is unreachable, continue locally.
        print(f"ClearML not available/reachable (continuing locally): {e}")


# Call this as early as possible, before heavy imports:
_maybe_init_clearml_and_queue()
_maybe_bootstrap_requirements()

# ----------------------------
# 1) Imports (after bootstrap)
# ----------------------------
import numpy as np  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize  # noqa: E402

from ot2_gym_wrapper import OT2GymWrapper  # noqa: E402


# ----------------------------
# 2) Training configuration
# ----------------------------
N_ENVS = int(os.getenv("N_ENVS", "8"))

MAX_STEPS = int(os.getenv("MAX_STEPS", "400"))
MAX_VEL = float(os.getenv("MAX_VEL", "0.4"))

# Curriculum stages (planar): 20mm -> 10mm
STAGE1 = dict(tolerance=float(os.getenv("STAGE1_TOL", "0.02")), timesteps=int(os.getenv("STAGE1_STEPS", "400000")))
STAGE2 = dict(tolerance=float(os.getenv("STAGE2_TOL", "0.01")), timesteps=int(os.getenv("STAGE2_STEPS", "1200000")))

# PPO hyperparams (can still override through env vars if you want)
PPO_LR = float(os.getenv("PPO_LR", "3e-4"))
PPO_N_STEPS = int(os.getenv("PPO_N_STEPS", "2048"))
PPO_BATCH = int(os.getenv("PPO_BATCH", "256"))
PPO_N_EPOCHS = int(os.getenv("PPO_EPOCHS", "10"))
PPO_GAMMA = float(os.getenv("PPO_GAMMA", "0.99"))
PPO_GAE_LAMBDA = float(os.getenv("PPO_GAE_LAMBDA", "0.95"))
PPO_ENT_COEF = float(os.getenv("PPO_ENT_COEF", "0.0"))
PPO_VF_COEF = float(os.getenv("PPO_VF_COEF", "0.5"))
PPO_CLIP_RANGE = float(os.getenv("PPO_CLIP_RANGE", "0.2"))

SEED = int(os.getenv("SEED", "0"))

# Output folders
ROOT_OUT = Path(os.getenv("OUT_DIR", "./outputs_local")).resolve()
MODELS_DIR = ROOT_OUT / "models"
BEST_DIR = ROOT_OUT / "best_models"
LOGS_DIR = ROOT_OUT / "logs"
ROOT_OUT.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
BEST_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def make_env(tolerance: float, rank: int):
    def _init():
        env = OT2GymWrapper(
            target_tolerance=tolerance,
            max_steps=MAX_STEPS,
            max_vel=MAX_VEL,
            render_mode=None,
        )
        env.reset(seed=SEED + rank)
        return env
    return _init


def build_vec_env(tolerance: float):
    env_fns = [make_env(tolerance, i) for i in range(N_ENVS)]
    venv = SubprocVecEnv(env_fns, start_method="spawn")
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return venv


def print_banner():
    print("=" * 70)
    print("RL TRAINING (PPO) - PLANAR CURRICULUM 20mm -> 10mm")
    print("=" * 70)
    print(f"Device: {'cuda/mps(if available)' if False else 'cpu'} | N_ENVS={N_ENVS} | MAX_VEL={MAX_VEL} | MAX_STEPS={MAX_STEPS}")
    print(f"Stage1: tol={STAGE1['tolerance']} for {STAGE1['timesteps']:,} steps")
    print(f"Stage2: tol={STAGE2['tolerance']} for {STAGE2['timesteps']:,} steps")
    print(f"Outputs: {ROOT_OUT}")
    print("=" * 70)


def main():
    print_banner()
    start_time = datetime.now()

    # ---- Stage 1 ----
    train_env = build_vec_env(STAGE1["tolerance"])
    eval_env = build_vec_env(STAGE1["tolerance"])

    # Share normalization statistics for evaluation (use same running stats)
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    eval_env.training = False

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(MODELS_DIR),
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(BEST_DIR),
        log_path=str(LOGS_DIR),
        eval_freq=50_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=PPO_LR,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        ent_coef=PPO_ENT_COEF,
        vf_coef=PPO_VF_COEF,
        clip_range=PPO_CLIP_RANGE,
        seed=SEED,
    )

    if STAGE1["timesteps"] > 0:
        model.learn(
            total_timesteps=STAGE1["timesteps"],
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )

    # ---- Stage 2 (transfer VecNormalize stats) ----
    if STAGE2["timesteps"] > 0:
        train_env_2 = build_vec_env(STAGE2["tolerance"])
        eval_env_2 = build_vec_env(STAGE2["tolerance"])

        # Transfer running stats from stage1 normalize to stage2 normalize
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
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        )

        model.learn(
            total_timesteps=STAGE2["timesteps"],
            callback=[checkpoint_callback, eval_callback_2],
            progress_bar=True,
        )

    # Save final model + VecNormalize stats
    final_model_path = MODELS_DIR / "final_model.zip"
    model.save(str(final_model_path))
    # Save vecnormalize (stage2 env if used else stage1)
    vec_path = MODELS_DIR / "vecnormalize.pkl"
    (model.get_env()).save(str(vec_path))

    elapsed = datetime.now() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Time elapsed: {elapsed}")
    print(f"Saved model: {final_model_path}")
    print(f"Saved VecNormalize: {vec_path}")
    print(f"Best model dir: {BEST_DIR}")
    print(f"Logs dir: {LOGS_DIR}")
    print("=" * 70)

    # Clean up
    try:
        train_env.close()
        eval_env.close()
    except Exception:
        pass


if __name__ == "__main__":
    # Ensure spawn for macOS stability with SubprocVecEnv
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
