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
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
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


# ─────────────────────────────────────────────
# STEP 2: Set Up the Atari Environment
# ─────────────────────────────────────────────
def make_atari_env():
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

    return env


# ─────────────────────────────────────────────
# STEP 3: Create the DQN Agent
# ─────────────────────────────────────────────
def create_agent(env):
    """
    Creates a DQN agent with our chosen hyperparameters.

    What is epsilon-greedy exploration?
    At first, the agent takes random actions (explores) to discover rewards.
    Over time, it takes the best known action more often (exploits).
    Epsilon controls the balance: high epsilon = more random, low = more greedy.
    """
    agent = DQN(
        policy=POLICY,
        env=env,
        learning_rate=HYPERPARAMS["learning_rate"],
        gamma=HYPERPARAMS["gamma"],
        batch_size=HYPERPARAMS["batch_size"],
        exploration_fraction=HYPERPARAMS["exploration_fraction"],
        exploration_final_eps=HYPERPARAMS["exploration_final_eps"],
        buffer_size=HYPERPARAMS["buffer_size"],
        learning_starts=HYPERPARAMS["learning_starts"],
        train_freq=HYPERPARAMS["train_freq"],
        target_update_interval=HYPERPARAMS["target_update_interval"],
        optimize_memory_usage=True,  # Saves RAM (important on Kaggle)
        verbose=1,                   # Print training progress to screen
        tensorboard_log="./logs/",   # Save training curves (optional)
    )
    return agent


# ─────────────────────────────────────────────
# STEP 4: Train the Agent
# ─────────────────────────────────────────────
def train():
    print("=" * 60)
    print(f"  Training Experiment #{EXPERIMENT_NUMBER}")
    print(f"  Policy: {POLICY}")
    print(f"  Hyperparameters:")
    for key, val in HYPERPARAMS.items():
        print(f"    {key}: {val}")
    print("=" * 60)

    # Create environment
    env = make_atari_env()
    eval_env = make_atari_env()  # A separate environment just for evaluation

    # Create the DQN agent
    agent = create_agent(env)

    # EvalCallback: Every 50,000 steps, test the agent for 5 episodes
    # and automatically save the best model found so far
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/experiment_{EXPERIMENT_NUMBER}/",
        log_path=f"./logs/experiment_{EXPERIMENT_NUMBER}/",
        eval_freq=50_000,       # Evaluate every 50k steps
        n_eval_episodes=5,      # Run 5 test episodes each time
        deterministic=True,     # Use greedy policy during evaluation
        render=False,
    )

    # Train the agent!
    # This is where the magic happens. The agent plays Pong, gets rewards,
    # and keeps improving its strategy.
    agent.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True,      # Shows a progress bar in terminal/Kaggle
    )

    # Save the final trained model
    os.makedirs("./models", exist_ok=True)
    save_path = f"./models/dqn_model_exp{EXPERIMENT_NUMBER}"
    agent.save(save_path)
    print(f"\n✅ Model saved to: {save_path}.zip")

    # Also save as the "best" model for play.py to use
    agent.save("./models/dqn_model")
    print("✅ Also saved as: ./models/dqn_model.zip")

    env.close()
    eval_env.close()


# ─────────────────────────────────────────────
# STEP 5: Run Training
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train()
