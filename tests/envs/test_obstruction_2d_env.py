"""Tests for obstruction_2d_env.py."""

import gymnasium as gym
import numpy as np

from geom2drobotenvs import register_all_environments
from geom2drobotenvs.envs.obstruction_2d_env import (
    Obstruction2DEnv,
    Obstruction2DEnvSpec,
)


def test_obstruction_2d_env_creation():
    """Tests creation of Obstruction2DEnv()."""
    register_all_environments()
    env = gym.make("geom2drobotenvs/Obstruction2D-v0")
    assert isinstance(env.unwrapped, Obstruction2DEnv)


# TODO add test for reset
