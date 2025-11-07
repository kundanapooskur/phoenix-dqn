import argparse
import os
import random
from typing import Tuple

import ale_py  # Register ALE environments
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from rl.agents.dqn_agent import DQNAgent, DQNConfig
from rl.replay_buffer import ReplayBuffer
from rl.utils.preprocessing import FrameStack
from rl.utils.schedules import ExponentialEpsilon, BoltzmannExplorer


def make_env(env_id: str, seed: int) -> gym.Env:
    env = gym.make(env_id, render_mode=None)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="ALE/Phoenix-v5")
    parser.add_argument("--total_episodes", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    # Prefer CUDA, then Apple MPS, else CPU
    default_device = (
        "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--train_dir", type=str, default="checkpoints/phoenix_dqn")
    parser.add_argument("--buffer_capacity", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target_update_interval", type=int, default=10_000)
    parser.add_argument("--start_learn_after", type=int, default=50_000)
    parser.add_argument("--train_freq", type=int, default=4)
    parser.add_argument("--exploration", type=str, choices=["epsilon", "boltzmann"], default="epsilon")
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay_rate", type=float, default=0.9995)
    parser.add_argument("--boltzmann_start_temp", type=float, default=1.0)
    parser.add_argument("--boltzmann_end_temp", type=float, default=0.1)
    parser.add_argument("--boltzmann_decay_steps", type=int, default=500_000)
    args = parser.parse_args()

    os.makedirs(args.train_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("runs", os.path.basename(args.train_dir)))

    set_seed(args.seed)
    env = make_env(args.env_id, args.seed)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    num_actions = env.action_space.n
    frame_stack = FrameStack(num_frames=4, height=84, width=84)

    agent = DQNAgent(
        DQNConfig(
            num_actions=num_actions,
            num_frames=4,
            gamma=args.gamma,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            target_update_interval=args.target_update_interval,
            device=args.device,
        )
    )

    replay = ReplayBuffer(capacity=args.buffer_capacity, state_shape=(4, 84, 84), dtype=np.uint8)

    epsilon_sched = ExponentialEpsilon(start=args.epsilon_start, end=args.epsilon_end, decay_rate=args.epsilon_decay_rate)
    boltz_explorer = BoltzmannExplorer(
        start_temperature=args.boltzmann_start_temp,
        end_temperature=args.boltzmann_end_temp,
        decay_steps=args.boltzmann_decay_steps,
    )

    global_step = 0
    best_avg_reward = -float("inf")
    best_path = os.path.join(args.train_dir, "best.pt")

    for episode in trange(args.total_episodes, desc="Training"):
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        done = False
        total_reward = 0.0

        for step in range(args.max_steps):
            if args.exploration == "epsilon":
                eps = epsilon_sched.value(global_step)
                action = agent.act_epsilon_greedy(state, eps)
                writer.add_scalar("exploration/epsilon", eps, global_step)
            else:
                temp = boltz_explorer.temperature(global_step)
                action = agent.act_boltzmann(state, temp)
                writer.add_scalar("exploration/temperature", temp, global_step)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = frame_stack.append(next_obs)
            done = terminated or truncated

            clipped_reward = float(np.clip(reward, -1.0, 1.0))
            replay.add(state, action, clipped_reward, next_state, done)

            state = next_state
            total_reward += reward
            global_step += 1

            if (len(replay) >= args.start_learn_after) and (global_step % args.train_freq == 0):
                batch = replay.sample(args.batch_size)
                loss = agent.learn(batch)
                writer.add_scalar("train/loss", loss, global_step)

            if done:
                break

        writer.add_scalar("episode/return", total_reward, episode)
        writer.add_scalar("episode/steps", step + 1, episode)

        # Simple evaluation every 20 episodes for model selection
        if (episode + 1) % 20 == 0:
            eval_returns = []
            for _ in range(5):
                eo, _ = env.reset()
                es = frame_stack.reset(eo)
                er = 0.0
                for _ in range(5_000):
                    a = agent.act_epsilon_greedy(es, epsilon=0.01)
                    eo2, r, t, tr, _ = env.step(a)
                    es = frame_stack.append(eo2)
                    er += r
                    if t or tr:
                        break
                eval_returns.append(er)
            avg_r = float(np.mean(eval_returns))
            writer.add_scalar("eval/avg_return", avg_r, episode)
            if avg_r > best_avg_reward:
                best_avg_reward = avg_r
                agent.save(best_path)

    # Save final
    agent.save(os.path.join(args.train_dir, "final.pt"))
    env.close()
    writer.close()


if __name__ == "__main__":
    main()


