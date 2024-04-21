"""Tests for shelf_world.py."""

from relational_structs.spaces import ObjectCentricStateSpace
from relational_structs.structs import State

from geom2drobotenvs.envs import ShelfWorldEnv
from geom2drobotenvs.object_types import CRVRobotType, RectangleType


def test_shelf_world_env():
    """Tests for ShelfWorldEnv()."""
    env = ShelfWorldEnv()
    assert isinstance(env.observation_space, ObjectCentricStateSpace)
    assert env.observation_space.types == {CRVRobotType, RectangleType}
    assert env.action_space.shape == (5,)
    obs, info = env.reset(seed=123)
    assert info == {}
    assert isinstance(obs, State)
