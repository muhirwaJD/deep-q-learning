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

## Project Structure
| File | Purpose |
|------|---------|
| `train.py` | Train the DQN agent locally |
| `play.py` | Load the trained model and play |
| `train_kaggle.ipynb` | Kaggle notebook for training on GPU |
| `requirements.txt` | Required Python packages |
| `models/dqn_model.zip` | Saved trained model (best experiment) |

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Training (on Kaggle — Recommended)
1. Upload `train_kaggle.ipynb` to [kaggle.com](https://kaggle.com)
2. Enable GPU: **Settings → Accelerator → GPU P100**
3. Run all cells
4. Download `dqn_model.zip` from the output panel
5. Place the model in `./models/dqn_model.zip`

### Training (Local)
```bash
python train.py
```
> **Note**: Training locally without a GPU will be very slow. Kaggle is recommended.

### Playing (Local)
```bash
python play.py
```
> Set `RENDER = True` inside `play.py` to see the game window on your screen.

---

## Policy Comparison: MLP vs CNN

| Policy | Description | Performance on Pong |
|--------|-------------|---------------------|
| **CnnPolicy** | Looks at raw game pixels using convolutional layers | ✅ Much better — designed for visual input |
| **MlpPolicy** | Uses flat/flattened input | ❌ Worse — not suited for image data |

**Conclusion**: For Atari games (visual input), **CnnPolicy** is the right choice. The convolutional layers detect spatial patterns like ball position and paddle location from raw pixels, which an MLP cannot do efficiently due to the lack of spatial feature extraction.

---

## Hyperparameter Tuning Results

> **Pong scoring**: +21 = perfect win, 0 = even match, -21 = total loss

Each group member experimented with 10 different hyperparameter configurations. The following table documents the results:

### Member: <!-- TODO: Replace with your name -->

| # | lr | gamma | batch | eps_start | eps_end | eps_frac | Mean Reward | Notes |
|---|-----|-------|-------|-----------|---------|----------|-------------|-------|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | <!-- TODO --> | Baseline configuration |
| 2 | 1e-3 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | <!-- TODO --> | Higher LR — tests learning speed |
| 3 | 5e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | <!-- TODO --> | Lower LR — tests stability |
| 4 | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 0.10 | <!-- TODO --> | Lower gamma — less future focus |
| 5 | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 0.10 | <!-- TODO --> | Larger batch — more stable updates |
| 6 | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 0.10 | <!-- TODO --> | Smaller batch — noisier updates |
| 7 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | <!-- TODO --> | Less final exploration |
| 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.30 | <!-- TODO --> | More exploration time |
| 9 | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 0.10 | <!-- TODO --> | Short-sighted agent |
| 10 | 2e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.15 | <!-- TODO --> | Combined tuning |

### Hyperparameter Tuning Discussion

<!-- TODO: Fill this in after running experiments -->

**Learning Rate (`lr`)**: 
- Higher learning rates (e.g., 1e-3) led to ___. Lower learning rates (e.g., 5e-5) led to ___.
- The best learning rate was ___ because ___.

**Gamma (Discount Factor)**:
- A gamma of 0.99 (default) ___, while reducing it to 0.90 caused ___.
- This makes sense because in Pong, future rewards (scoring points) ___.

**Batch Size**:
- Larger batches (64) resulted in ___, while smaller batches (16) ___.
- The optimal batch size was ___ because ___.

**Exploration (Epsilon)**:
- More exploration time (eps_frac=0.30) led to ___.
- Less final exploration (eps_end=0.01) caused ___.
- The balance between exploration and exploitation ___.

**Best Configuration**: Experiment #___ performed best with a mean reward of ___ because ___.

---

## Hyperparameter Reference

| Hyperparameter | What It Controls |
|---|---|
| **Learning Rate (`lr`)** | Step size for weight updates. Too high = unstable. Too low = slow convergence. |
| **Gamma (`γ`)** | Discount factor for future rewards. 0.99 = values long-term rewards. Lower = more short-sighted. |
| **Batch Size** | Number of experiences sampled per training step. Larger = more stable but slower. |
| **Epsilon Start** | Always 1.0 — agent starts by exploring 100% randomly. |
| **Epsilon End (`eps_end`)** | Final exploration rate. 0.05 = 5% random actions at the end of exploration phase. |
| **Epsilon Fraction (`eps_frac`)** | Fraction of total training steps over which epsilon decays from start to end. |

---

## Gameplay Demo

<!-- TODO: Replace with your actual gameplay GIF after training -->
<!-- Run record_gif("gameplay.gif") in play.py or use the Kaggle notebook Step 8 -->
![Agent playing Pong](gameplay.gif)

---

## Group Members & Contributions

<!-- TODO: Fill in your group member details -->
| Member | Contribution |
|--------|-------------|
| Member 1 | Hyperparameter experiments 1-10, ___ |
| Member 2 | Hyperparameter experiments 1-10, ___ |
| Member 3 | Hyperparameter experiments 1-10, ___ |
