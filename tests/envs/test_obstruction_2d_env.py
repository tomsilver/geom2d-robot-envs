"""Tests for obstruction_2d_env.py."""

import gymnasium as gym
import numpy as np

from geom2drobotenvs import register_all_environments
from geom2drobotenvs.envs.obstruction_2d_env import (
    Obstruction2DEnv,
    Obstruction2DEnvSpec,
    TargetBlockType,
    TargetSurfaceType,
)
from geom2drobotenvs.structs import SE2Pose


def test_obstruction_2d_env_creation():
    """Tests creation of Obstruction2DEnv()."""
    register_all_environments()
    env = gym.make("geom2drobotenvs/Obstruction2D-v0")
    assert isinstance(env.unwrapped, Obstruction2DEnv)


def test_obstruction_2d_env_reset():
    """Tests that resetting the Obstruction2DEnv() does not crash."""
    register_all_environments()
    env = gym.make("geom2drobotenvs/Obstruction2D-v0")
    for seed in range(50):
        env.reset(seed=seed)


def test_successful_pick_place_no_obstructions():
    """Tests a hardcoded plan in the case where there are no obstructions."""
    init_spec = Obstruction2DEnvSpec()
    robot_init_pose = SE2Pose(
        (init_spec.world_max_x + init_spec.world_min_x) / 2,
        init_spec.world_max_y - 1.5 * init_spec.robot_base_radius,
        -np.pi / 2,
    )
    target_block_width = np.mean(init_spec.target_block_width_bounds)
    target_block_height = np.mean(init_spec.target_block_height_bounds)
    target_surface_width_addition = np.mean(
        init_spec.target_surface_width_addition_bounds
    )
    target_surface_width = target_block_width + target_surface_width_addition
    target_surface_init_pose = SE2Pose(
        (init_spec.world_max_x + init_spec.world_min_x) / 2 - target_surface_width / 2,
        init_spec.target_surface_init_pose_bounds[0].y,
        init_spec.target_surface_init_pose_bounds[0].theta,
    )
    target_block_init_pose = SE2Pose(
        init_spec.world_min_x + 0.1 * (init_spec.world_max_x - init_spec.world_min_x),
        init_spec.target_block_init_pose_bounds[0].y,
        init_spec.target_block_init_pose_bounds[0].theta,
    )
    spec = Obstruction2DEnvSpec(
        robot_init_pose_bounds=(robot_init_pose, robot_init_pose),
        target_block_width_bounds=(target_block_width, target_block_width),
        target_block_height_bounds=(target_block_height, target_block_height),
        target_block_init_pose_bounds=(target_block_init_pose, target_block_init_pose),
        target_surface_init_pose_bounds=(
            target_surface_init_pose,
            target_surface_init_pose,
        ),
    )
    env = Obstruction2DEnv(num_obstructions=0, spec=spec)

    # TODO remove
    # obs, _ = env.reset(seed=123)
    # import imageio.v2 as iio
    # iio.imsave("debug.png", env.render())
