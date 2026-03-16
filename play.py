"""
play.py — Load and Play with the Trained DQN Agent
===================================================
This script loads your trained model and lets the agent play Pong.

What happens here:
  1. Load the saved DQN model (dqn_model.zip)
  2. Create the Pong game environment
  3. The agent looks at the screen and picks the best action each step
  4. This is called "Greedy Policy" — always pick the action with the
     highest Q-value (no more random exploration, just using what it learned)
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


# ─────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────

# Path to your saved model
MODEL_PATH = "./models/dqn_model"

# How many episodes to play
NUM_EPISODES = 5

# Set to True to show the game window on your screen (local PC with display)
# Set to False on headless servers (Kaggle, WSL without display, SSH, etc.)
# TIP: If you get a display error, set this to False
RENDER = False


# ─────────────────────────────────────────────
# Set Up the Environment
# ─────────────────────────────────────────────
def make_play_env(render=False):
    """
    Creates the Pong environment for playing (with or without display).
    Same wrappers as training to make sure the agent sees the same input.
    """
    render_mode = "human" if render else "rgb_array"

    def _make():
        env = gym.make("ALE/Pong-v5", render_mode=render_mode)
        env = AtariWrapper(env)
        return env

    env = DummyVecEnv([_make])
    env = VecFrameStack(env, n_stack=4)
    return env


# ─────────────────────────────────────────────
# Play Function
# ─────────────────────────────────────────────
def play():
    print("=" * 50)
    print("  Loading trained DQN model...")
    print("=" * 50)

    # Load the trained model
    # Note: We pass the env so SB3 knows what action/observation space to use
    env = make_play_env(render=RENDER)
    model = DQN.load(MODEL_PATH, env=env)
    print(f"✅ Model loaded from: {MODEL_PATH}.zip")

    # Track scores across episodes
    episode_rewards = []

    print(f"\n🎮 Playing {NUM_EPISODES} episodes...\n")

    for episode in range(1, NUM_EPISODES + 1):
        # Reset: start a fresh game
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # ─────────────────────────────────────────
            # GREEDY POLICY: Pick the best action
            # ─────────────────────────────────────────
            # deterministic=True means NO random exploration.
            # The agent always picks the action with the highest Q-value.
            # This is like a student who has finished studying and is
            # now just using what they know (no more guessing).
            action, _ = model.predict(obs, deterministic=True)

            # Take the action in the environment
            obs, reward, done, info = env.step(action)

            total_reward += reward[0]  # reward is an array in VecEnv
            step_count += 1

        episode_rewards.append(total_reward)
        print(f"  Episode {episode}/{NUM_EPISODES} | "
              f"Reward: {total_reward:.1f} | "
              f"Steps: {step_count}")

    # ─────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Results Summary")
    print("=" * 50)
    print(f"  Average Reward : {np.mean(episode_rewards):.2f}")
    print(f"  Best Reward    : {np.max(episode_rewards):.2f}")
    print(f"  Worst Reward   : {np.min(episode_rewards):.2f}")
    print("=" * 50)

    # In Pong:
    # +21 = perfect win (you scored 21, opponent scored 0)
    # -21 = total loss (opponent scored 21)
    # ~0  = competitive game
    if np.mean(episode_rewards) > 0:
        print("\n🏆 Agent is winning! Great training result.")
    elif np.mean(episode_rewards) > -10:
        print("\n📈 Agent is competitive. Try more training steps.")
    else:
        print("\n📉 Agent needs more training. Increase TOTAL_TIMESTEPS.")

    env.close()


# ─────────────────────────────────────────────
# Save Gameplay Frames as a GIF (for README/submission)
# ─────────────────────────────────────────────
def record_gif(output_path="gameplay.gif", num_steps=500):
    """
    Records the agent playing and saves it as a GIF.
    Use this to create the video for your README submission.
    """
    import imageio

    print("\n📹 Recording gameplay GIF...")

    def _make():
        env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
        env = AtariWrapper(env)
        return env

    raw_env = DummyVecEnv([_make])
    env = VecFrameStack(raw_env, n_stack=4)

    model = DQN.load(MODEL_PATH, env=env)

    frames = []
    obs = env.reset()

    for _ in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

        # Get the raw RGB frame
        frame = raw_env.envs[0].env.render()
        if frame is not None:
            frames.append(frame)

        if done[0]:
            obs = env.reset()

    env.close()

    # Save as GIF
    imageio.mimsave(output_path, frames, fps=30)
    print(f"✅ Gameplay saved to: {output_path}")


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    play()

    # Save a gameplay GIF for your README submission
    print("\n📹 Now recording gameplay GIF for README...")
    record_gif("gameplay.gif")
