"""Create a GIF that shows different initial states after calling reset()."""

import argparse
from pathlib import Path

import gymnasium as gym
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from geom2drobotenvs import register_all_environments


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str)
    parser.add_argument("--init_seed", required=False, type=int, default=0)
    parser.add_argument("--num", required=False, type=int, default=50)
    parser.add_argument("--fps", required=False, type=int, default=10)
    parser.add_argument("--outdir", required=False, type=Path, default="reset_gifs")
    args = parser.parse_args()
    register_all_environments()
    env = gym.make(args.env)
    imgs = []
    for seed in range(args.init_seed, args.init_seed + args.num):
        env.reset(seed=seed)
        img = env.render()
        imgs.append(img)
    env.close()
    args.outdir.mkdir(exist_ok=True)
    env_name = args.env.replace("/", "-")
    outfile = args.outdir / f"{env_name}.gif"
    clip = ImageSequenceClip(imgs, fps=args.fps)
    clip.write_gif(outfile)
    print(f"Saved GIF to {outfile}")


if __name__ == "__main__":
    _main()
