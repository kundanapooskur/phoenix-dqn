import argparse
import os

import ale_py  # Register ALE environments
import gymnasium as gym
import numpy as np
import torch

from rl.agents.dqn_agent import DQNAgent, DQNConfig
from rl.utils.preprocessing import FrameStack


def make_env(env_id: str, seed: int, video_dir: str | None):
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda e: True)
    else:
        env = gym.make(env_id, render_mode=None)
    env.reset(seed=seed)
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="ALE/Phoenix-v5")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    default_device = (
        "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--video_dir", type=str, default="videos/phoenix")
    args = parser.parse_args()

    env = make_env(args.env_id, args.seed, args.video_dir)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    num_actions = env.action_space.n
    frame_stack = FrameStack(num_frames=4, height=84, width=84)

    agent = DQNAgent(DQNConfig(num_actions=num_actions, num_frames=4, device=args.device))
    agent.load(args.checkpoint, map_location=args.device)

    returns = []
    steps = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        done = False
        ep_return = 0.0
        ep_steps = 0
        while not done:
            action = agent.act_epsilon_greedy(state, epsilon=0.01)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            state = frame_stack.append(next_obs)
            ep_return += reward
            ep_steps += 1
            done = terminated or truncated
        returns.append(ep_return)
        steps.append(ep_steps)
        print(f"Episode {ep+1}: return={ep_return:.2f}, steps={ep_steps}")

    print(f"Average return over {args.episodes} episodes: {np.mean(returns):.2f}")
    print(f"Average steps per episode: {np.mean(steps):.1f}")
    env.close()


if __name__ == "__main__":
    main()


