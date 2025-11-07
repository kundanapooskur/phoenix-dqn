import argparse
import os

import ale_py  # Register ALE environments
import gymnasium as gym


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="ALE/Phoenix-v5")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video_dir", type=str, default="videos/random_demo")
    args = parser.parse_args()

    os.makedirs(args.video_dir, exist_ok=True)
    env = gym.make(args.env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=args.video_dir, episode_trigger=lambda e: True)
    env.reset(seed=args.seed)

    obs, info = env.reset()
    done = False
    step_count = 0
    while step_count < args.steps and not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

    env.close()
    print(f"Saved video(s) to: {args.video_dir}")


if __name__ == "__main__":
    main()



