"""Tests for shelf_world.py."""

import numpy as np
from gym.spaces import Box
from gym.wrappers.record_video import RecordVideo
from relational_structs.spaces import ObjectCentricStateSpace
from relational_structs.structs import State
from relational_structs.utils import create_state_from_dict

from geom2drobotenvs.envs import ShelfWorldEnv
from geom2drobotenvs.object_types import CRVRobotType, RectangleType


def test_shelf_world_env():
    """Basic tests for ShelfWorldEnv()."""
    env = ShelfWorldEnv()
    assert isinstance(env.observation_space, ObjectCentricStateSpace)
    assert env.observation_space.types == {CRVRobotType, RectangleType}
    assert env.action_space.shape == (5,)
    obs, info = env.reset(seed=123)
    assert info == {}
    assert isinstance(obs, State)


def test_shelf_world_robot_moves():
    """Test basic movements of the robot in ShelfWorldEnv()."""
    env = ShelfWorldEnv()

    # Uncomment to record videos.
    # TODO: comment out.
    env = RecordVideo(env, "unit_test_videos")

    # Set up an initial scene with just the robot.
    robot = CRVRobotType("robot")
    init_state_dict = {
        robot: {
            "x": 5.0,  # center of room
            "y": 5.0,
            "theta": 0.0,  # facing right
            "base_radius": 0.5,
            "arm_joint": 0.5,  # arm is fully retracted
            "vacuum": 0.0,  # vacuum is off
        }
    }
    init_state = create_state_from_dict(init_state_dict)
    obs, _ = env.reset(seed=123, options={"init_state": init_state})
    assert np.isclose(obs.get(robot, "theta"), 0.0)  # sanity check

    # Move all the way to the right. The number is chosen to be gratuitous.
    action_space = env.action_space
    assert isinstance(action_space, Box)
    right_action = np.zeros_like(action_space.high)
    right_action[0] = action_space.high[0]
    for _ in range(50):
        obs, _, _, _, _ = env.step(right_action)

    # Finish.
    env.close()
