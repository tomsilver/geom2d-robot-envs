"""Quick demo script to try out environments with random actions."""

import argparse
import geom2drobotenvs
import gym
import imageio.v2 as iio

def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str)
    parser.add_argument("--seed", required=False, type=int, default=0)
    parser.add_argument("--steps", required=False, type=int, default=32)
    parser.add_argument("--outfile", required=False, type=str, default="video.mp4")
    args = parser.parse_args()
    imgs = []
    env = gym.make(args.env)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    imgs.append(env.render())
    for _ in range(args.steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        imgs.append(env.render())
        if terminated or truncated:
            print("WARNING: terminating early.")
            break
    iio.mimsave(args.outfile, imgs)
    print(f"Wrote out to {args.outfile}")
    

if __name__ == "__main__":
    _main()
