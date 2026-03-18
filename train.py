"""
train.py — Deep Q-Network (DQN) Training Script
================================================
This script trains a DQN agent to play Pong (an Atari game).

What is DQN?
  - DQN is a type of Reinforcement Learning algorithm.
  - The agent watches the game screen (pixels) and learns which actions
    give the most reward over time.
  - It uses a neural network (CNN) to "look" at the screen and decide what to do.

How Reinforcement Learning works (simple version):
  1. Agent sees the current game screen (state)
  2. Agent picks an action (move paddle up, down, or stay)
  3. Game gives back a reward (+1 for scoring, -1 for opponent scoring)
  4. Agent learns from this experience and improves over time
"""

import os
import csv
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import ale_py
import gymnasium as gym

# Register ALE Atari environments with Gymnasium
gym.register_envs(ale_py)

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np


# ─────────────────────────────────────────────
# STEP 1: Choose Your Hyperparameters
# ─────────────────────────────────────────────
# These are the settings that control how the agent learns.
# Change these values to do your 10 experiments.

HYPERPARAMS = {
    "learning_rate": 1e-4,      # How fast the agent updates its knowledge
                                 # Too high → unstable learning
                                 # Too low → very slow learning

    "gamma": 0.99,               # Discount factor: how much the agent values
                                 # future rewards vs immediate rewards
                                 # 0 = only cares about now, 1 = cares about all future

    "batch_size": 32,            # How many past experiences to learn from at once
                                 # Larger = more stable but slower

    "exploration_fraction": 0.1, # What fraction of training to spend exploring
                                 # (epsilon goes from 1.0 → 0.05 during this fraction)

    "exploration_final_eps": 0.05, # Final exploration rate (5% random actions at the end)

    "buffer_size": 100_000,      # How many past experiences to remember
                                 # Think of this as the agent's "memory size"

    "learning_starts": 10_000,   # Don't start learning until this many steps
                                 # (fill the memory first)

    "train_freq": 4,             # Learn every 4 steps (not every single step)

    "target_update_interval": 1000, # How often to update the "target" network
                                     # (a stable copy used to calculate target Q-values)
}

# How many steps to train for.
# On Kaggle: use 1_000_000 for a proper run (~30 min on GPU)
# For a quick test: use 100_000
TOTAL_TIMESTEPS = 1_000_000

# Which experiment number is this? (1–10)
# Change this each time you run with different hyperparameters
EXPERIMENT_NUMBER = 1

# Policy type: "CnnPolicy" or "MlpPolicy"
# CnnPolicy = looks at the raw game pixels (recommended for Atari)
# MlpPolicy = uses flattened features (worse for visual games)
POLICY = "CnnPolicy"


@dataclass(frozen=True)
class ExperimentConfig:
    number: int
    policy: str
    hyperparams: Dict[str, Any]
    total_timesteps: int = TOTAL_TIMESTEPS
    eval_freq: int = 50_000
    n_eval_episodes: int = 5


def _utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def member2_experiment_hyperparams() -> Dict[int, Dict[str, Any]]:
    """
    Member 2: 10 hyperparameter configurations (distinct combinations).
    These match the table in README under "Member 2".
    """
    base = dict(HYPERPARAMS)
    return {
        1: {**base, "learning_rate": 1e-4, "gamma": 0.99, "batch_size": 32, "exploration_fraction": 0.10, "exploration_final_eps": 0.05},
        2: {**base, "learning_rate": 5e-4, "gamma": 0.99, "batch_size": 32, "exploration_fraction": 0.10, "exploration_final_eps": 0.05},
        3: {**base, "learning_rate": 1e-5, "gamma": 0.99, "batch_size": 32, "exploration_fraction": 0.10, "exploration_final_eps": 0.05},
        4: {**base, "learning_rate": 1e-4, "gamma": 0.999, "batch_size": 32, "exploration_fraction": 0.10, "exploration_final_eps": 0.05},
        5: {**base, "learning_rate": 1e-4, "gamma": 0.99, "batch_size": 128, "exploration_fraction": 0.10, "exploration_final_eps": 0.05},
        6: {**base, "learning_rate": 1e-4, "gamma": 0.99, "batch_size": 32, "exploration_fraction": 0.10, "exploration_final_eps": 0.10},
        7: {**base, "learning_rate": 1e-4, "gamma": 0.99, "batch_size": 32, "exploration_fraction": 0.05, "exploration_final_eps": 0.05},
        8: {**base, "learning_rate": 1e-4, "gamma": 0.99, "batch_size": 32, "exploration_fraction": 0.50, "exploration_final_eps": 0.05},
        9: {**base, "learning_rate": 5e-4, "gamma": 0.95, "batch_size": 64, "exploration_fraction": 0.10, "exploration_final_eps": 0.05},
        10: {**base, "learning_rate": 1e-4, "gamma": 0.99, "batch_size": 32, "exploration_fraction": 0.20, "exploration_final_eps": 0.02},
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN on ALE/Pong-v5 with reproducible experiment logging.")
    p.add_argument("--member2", action="store_true", help="Use Member 2's 10 experiment configs from README.")
    p.add_argument("--exp", type=int, default=None, help="Experiment number (1-10 for --member2).")
    p.add_argument("--policy", type=str, default=None, choices=["cnn", "mlp"], help="Override policy: cnn or mlp.")
    p.add_argument("--timesteps", type=int, default=None, help="Override TOTAL_TIMESTEPS for a run.")
    return p.parse_args()


def _policy_from_arg(arg: Optional[str]) -> Optional[str]:
    if arg is None:
        return None
    return "CnnPolicy" if arg.lower() == "cnn" else "MlpPolicy"


# ─────────────────────────────────────────────
# STEP 2: Set Up the Atari Environment
# ─────────────────────────────────────────────
def make_atari_env(monitor_log_path: Optional[str] = None):
    """
    Creates a properly configured Atari Pong environment.

    Why do we need wrappers?
    The raw Atari game has too many pixels and runs at 60fps.
    These wrappers make it easier for the agent to learn:
      - AtariWrapper: Converts color → grayscale, resizes to 84x84
      - VecFrameStack: Stacks 4 frames so the agent can sense movement
        (like how a flip book creates the illusion of movement)
    """
    def _make():
        env = gym.make("ALE/Pong-v5", render_mode=None)
        env = AtariWrapper(env)  # Grayscale + resize to 84x84
        return env

    # DummyVecEnv wraps it into a "vectorized" environment (SB3 requirement)
    env = DummyVecEnv([_make])

    # Stack 4 frames together so the agent can see motion
    # (one frame alone doesn't show if the ball is moving left or right)
    env = VecFrameStack(env, n_stack=4)

    # VecMonitor logs episode reward + length for reward trends/episode length.
    # This satisfies the rubric requirement for training logs.
    if monitor_log_path is not None:
        os.makedirs(os.path.dirname(monitor_log_path), exist_ok=True)
        env = VecMonitor(env, filename=monitor_log_path)

    return env


# ─────────────────────────────────────────────
# STEP 3: Create the DQN Agent
# ─────────────────────────────────────────────
def create_agent(env, policy: str, hyperparams: Dict[str, Any]):
    """
    Creates a DQN agent with our chosen hyperparameters.

    What is epsilon-greedy exploration?
    At first, the agent takes random actions (explores) to discover rewards.
    Over time, it takes the best known action more often (exploits).
    Epsilon controls the balance: high epsilon = more random, low = more greedy.
    """
    agent = DQN(
        policy=policy,
        env=env,
        learning_rate=hyperparams["learning_rate"],
        gamma=hyperparams["gamma"],
        batch_size=hyperparams["batch_size"],
        exploration_fraction=hyperparams["exploration_fraction"],
        exploration_final_eps=hyperparams["exploration_final_eps"],
        buffer_size=hyperparams["buffer_size"],
        learning_starts=hyperparams["learning_starts"],
        train_freq=hyperparams["train_freq"],
        target_update_interval=hyperparams["target_update_interval"],
        optimize_memory_usage=False, # FIXED: Set to False to avoid ValueError with handle_timeout_termination
        verbose=1,                   # Print training progress to screen
        tensorboard_log="./logs/",   # Save training curves (optional)
    )
    return agent


# ─────────────────────────────────────────────
# STEP 4: Train the Agent
# ─────────────────────────────────────────────
def _append_results_row(row: Dict[str, Any], csv_path: str) -> None:
    """Append one row to CSV. Safe for parallel runs (uses file lock on Unix)."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass  # Windows or no flock; single-writer still usually ok
        try:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        finally:
            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass


def _evaluate_agent(model: DQN, eval_env, n_eval_episodes: int) -> Tuple[float, float]:
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        return_episode_rewards=True,
    )
    return float(np.mean(episode_rewards)), float(np.mean(episode_lengths))


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    print("=" * 60)
    print(f"  Training Experiment #{cfg.number}")
    print(f"  Policy: {cfg.policy}")
    print("  Hyperparameters:")
    for key, val in cfg.hyperparams.items():
        print(f"    {key}: {val}")
    print(f"  Total timesteps: {cfg.total_timesteps}")
    print("=" * 60)

    monitor_train = f"./logs/experiment_{cfg.number}/train_monitor.csv"
    monitor_eval = f"./logs/experiment_{cfg.number}/eval_monitor.csv"

    env = make_atari_env(monitor_log_path=monitor_train)
    eval_env = make_atari_env(monitor_log_path=monitor_eval)

    agent = create_agent(env, policy=cfg.policy, hyperparams=cfg.hyperparams)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/experiment_{cfg.number}/",
        log_path=f"./logs/experiment_{cfg.number}/",
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    agent.learn(
        total_timesteps=cfg.total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    os.makedirs("./models", exist_ok=True)
    save_path = f"./models/dqn_model_exp{cfg.number}"
    agent.save(save_path)
    print(f"\n✅ Model saved to: {save_path}.zip")

    # Also save as the "default" model for play.py to use
    agent.save("./models/dqn_model")
    print("✅ Also saved as: ./models/dqn_model.zip")

    mean_reward, mean_ep_len = _evaluate_agent(agent, eval_env, n_eval_episodes=cfg.n_eval_episodes)
    print("\n" + "-" * 60)
    print(f"Evaluation (greedy, {cfg.n_eval_episodes} eps) | Mean reward: {mean_reward:.2f} | Mean episode length: {mean_ep_len:.1f}")
    print("-" * 60)

    row = {
        "timestamp_utc": _utc_timestamp(),
        "experiment": cfg.number,
        "policy": cfg.policy,
        "total_timesteps": cfg.total_timesteps,
        "mean_reward": mean_reward,
        "mean_episode_length": mean_ep_len,
        "learning_rate": cfg.hyperparams["learning_rate"],
        "gamma": cfg.hyperparams["gamma"],
        "batch_size": cfg.hyperparams["batch_size"],
        "exploration_fraction": cfg.hyperparams["exploration_fraction"],
        "exploration_final_eps": cfg.hyperparams["exploration_final_eps"],
        "buffer_size": cfg.hyperparams["buffer_size"],
        "learning_starts": cfg.hyperparams["learning_starts"],
        "train_freq": cfg.hyperparams["train_freq"],
        "target_update_interval": cfg.hyperparams["target_update_interval"],
    }
    _append_results_row(row, csv_path="./results/experiment_results.csv")

    env.close()
    eval_env.close()

    return row


def train():
    print("=" * 60)
    print(f"  Training Experiment #{EXPERIMENT_NUMBER}")
    print(f"  Policy: {POLICY}")
    print(f"  Hyperparameters:")
    for key, val in HYPERPARAMS.items():
        print(f"    {key}: {val}")
    print("=" * 60)

    cfg = ExperimentConfig(
        number=EXPERIMENT_NUMBER,
        policy=POLICY,
        hyperparams=HYPERPARAMS,
        total_timesteps=TOTAL_TIMESTEPS,
    )
    run_experiment(cfg)


# ─────────────────────────────────────────────
# STEP 5: Run Training
# ─────────────────────────────────────────────
if __name__ == "__main__":
    args = _parse_args()
    if args.member2:
        if args.exp is None or args.exp not in member2_experiment_hyperparams():
            raise SystemExit("For --member2, pass --exp 1..10 (example: python3 train.py --member2 --exp 1).")
        exp_num = int(args.exp)
        hp = member2_experiment_hyperparams()[exp_num]
        policy_override = _policy_from_arg(args.policy)
        cfg = ExperimentConfig(
            number=exp_num,
            policy=policy_override or POLICY,
            hyperparams=hp,
            total_timesteps=int(args.timesteps) if args.timesteps is not None else TOTAL_TIMESTEPS,
        )
        run_experiment(cfg)
    else:
        train()
