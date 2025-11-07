# Phoenix DQN (Deep Q-Learning) Atari Agent

This project implements a Deep Q-Network (DQN) agent that learns to play the Atari 2600 game Phoenix using Gymnasium and ALE-Py. It includes a full training pipeline, evaluation script with video recording, TensorBoard logging, and a report answering the assignment questions.

## Quick Start

### 1) Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Note: Atari ROMs are provided by `ale-py` and accepted automatically via `gymnasium[accept-rom-license]`.

### 2) Train (500 episodes on Phoenix)

```bash
python -m rl.train_dqn \
  --env_id ALE/Phoenix-v5 \
  --total_episodes 500 \
  --train_dir checkpoints/phoenix_dqn \
  --seed 42
```

TensorBoard:

```bash
tensorboard --logdir runs
```

### 3) Evaluate with Video

```bash
python -m rl.eval_dqn \
  --env_id ALE/Phoenix-v5 \
  --checkpoint checkpoints/phoenix_dqn/best.pt \
  --episodes 10 \
  --video_dir videos/phoenix
```

Videos will be saved in `videos/phoenix`. Ensure `ffmpeg` is installed for video encoding.

## Quick Visual (No Training)

Record a short random-play video just to see the Phoenix game rendering:

```bash
python -m rl.record_random --env_id ALE/Phoenix-v5 --steps 2000 --video_dir videos/random_demo
```

Open the resulting MP4 in `videos/random_demo`.

## Project Layout

```
rl/
  __init__.py
  train_dqn.py
  eval_dqn.py
  agents/
    __init__.py
    dqn_agent.py
  models/
    __init__.py
    dqn_cnn.py
  utils/
    __init__.py
    preprocessing.py
    schedules.py
  replay_buffer.py
docs/
  REPORT.md
requirements.txt
LICENSE
README.md
```

## Report and Grading

See `docs/REPORT.md` for baseline configuration, analysis, answers to conceptual questions, policy exploration variations, and RL vs LLM agent discussions.

## License and Attribution

This project is licensed under the MIT License (see `LICENSE`). Code was written for this assignment from scratch; where we follow standard DQN design patterns (experience replay, target network, frame stacking), these are based on the original DQN paper by Mnih et al. (2015). No third‑party code was copied into the repository.


# phoenix-dqn
