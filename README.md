# Deep Q-Learning — Atari Pong
**Formative 3 Assignment | ALU**

## What This Project Does
We trained a DQN (Deep Q-Network) agent to play **Pong** using Stable Baselines 3 and Gymnasium.

The agent learns by:
1. Watching the game screen (84×84 grayscale pixels, 4 frames stacked)
2. Trying different actions (move paddle up/down/stay)
3. Getting rewarded (+1 for scoring, -1 for opponent scoring)
4. Repeating this millions of times until it learns a good strategy

## Environment
**ALE/Pong-v5** — Classic Atari Pong
- **Actions**: 6 (NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE)
- **Observation**: 84×84 grayscale pixels, 4 frames stacked
- **Reward**: +1 (you score), -1 (opponent scores), 0 (otherwise)

## Files
| File | Purpose |
|------|---------|
| `train.py` | Train the DQN agent locally |
| `play.py` | Load the trained model and play |
| `train_kaggle.ipynb` | Kaggle notebook for training on GPU |
| `requirements.txt` | Required Python packages |
| `models/dqn_model.zip` | Saved trained model |

## How to Run

### Training (Kaggle)
1. Upload `train_kaggle.ipynb` to [kaggle.com](https://kaggle.com)
2. Enable GPU: **Settings → Accelerator → GPU P100**
3. Run all cells
4. Download `dqn_model.zip` from the output panel

### Playing (Local)
```bash
pip install -r requirements.txt
python play.py
```

## Policy Comparison: MLP vs CNN

| Policy | Description | Performance on Pong |
|--------|-------------|---------------------|
| **CnnPolicy** | Looks at raw game pixels using convolutional layers | ✅ Much better — designed for visual input |
| **MlpPolicy** | Uses flat/flattened input | ❌ Worse — not designed for image data |

**Conclusion**: For Atari games (visual input), CnnPolicy is always the right choice. The convolutional layers detect patterns like the ball position and paddle location from pixels, which MLP cannot do efficiently.

## Hyperparameter Tuning Results

> **In Pong**: +21 = perfect win, 0 = even match, -21 = total loss

| # | Member | lr | gamma | batch | eps_start | eps_end | eps_frac | Mean Reward | Notes |
|---|--------|----|-------|-------|-----------|---------|----------|-------------|-------|
| 1 | [Name] | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | - | Baseline |
| 2 | [Name] | 1e-3 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | - | Higher LR → unstable |
| 3 | [Name] | 5e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | - | Lower LR → slow |
| 4 | [Name] | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 0.10 | - | Lower gamma |
| 5 | [Name] | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 0.10 | - | Larger batch |
| 6 | [Name] | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 0.10 | - | Smaller batch |
| 7 | [Name] | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | - | Less final exploration |
| 8 | [Name] | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.30 | - | More exploration time |
| 9 | [Name] | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 0.10 | - | Short-sighted agent |
| 10 | [Name] | 2e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.15 | - | Best combo guess |

*(Fill in Mean Reward column after running each experiment on Kaggle)*

## Gameplay Demo
![Gameplay](gameplay.gif)

## What Each Hyperparameter Controls

| Hyperparameter | Simple Explanation |
|---|---|
| **Learning Rate** | How big each learning step is. Too high = chaotic. Too low = very slow. |
| **Gamma** | How much the agent values future rewards. 0.99 means it cares a lot about future points, not just immediate ones. |
| **Batch Size** | How many past experiences to learn from at once. Larger = more stable but slower. |
| **Epsilon Start** | Always 1.0 — agent starts by exploring randomly (100% random). |
| **Epsilon End** | Final % of random actions. 0.05 = 5% random at the end. |
| **Epsilon Fraction** | What fraction of training time is spent reducing epsilon (1.0 → 0.05). |
