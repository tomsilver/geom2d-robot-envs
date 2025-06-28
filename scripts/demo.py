"""Quick demo script to try out environments with random actions."""

import argparse

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# Needed to register environments for gym.make().
from geom2drobotenvs import register_all_environments


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str)
    parser.add_argument("--seed", required=False, type=int, default=0)
    parser.add_argument("--steps", required=False, type=int, default=32)
    parser.add_argument("--outdir", required=False, type=str, default="demo_videos")
    args = parser.parse_args()
    register_all_environments()
    env = gym.make(args.env)
    env = RecordVideo(env, args.outdir)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    for _ in range(args.steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            print("WARNING: terminating early.")
            break
    env.close()
    print(f"Saved video to {args.outdir}")


if __name__ == "__main__":
    _main()
