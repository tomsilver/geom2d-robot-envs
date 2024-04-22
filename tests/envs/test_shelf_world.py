"""Tests for shelf_world.py."""

import numpy as np
from gym.spaces import Box
from gym.wrappers.record_video import RecordVideo
from relational_structs.spaces import ObjectCentricStateSpace
from relational_structs.structs import State
from relational_structs.utils import create_state_from_dict

from geom2drobotenvs.envs import ShelfWorldEnv
from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.utils import (
    CRVRobotActionSpace,
    create_walls_from_world_boundaries,
)


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

    world_min_x = env.unwrapped._world_min_x
    world_min_y = env.unwrapped._world_min_y
    world_max_x = env.unwrapped._world_max_x
    world_max_y = env.unwrapped._world_max_y

    # Set up an initial scene with just the robot.
    robot = CRVRobotType("robot")
    init_state_dict = {
        robot: {
            "x": (world_min_x + world_max_x) / 2,
            "y": (world_min_y + world_max_y) / 2,
            "theta": 0.0,
            "base_radius": 0.5,
            "arm_joint": 0.5,
            "vacuum": 0.0,
        }
    }
    assert isinstance(env.action_space, CRVRobotActionSpace)
    min_dx, min_dy = env.action_space.low[:2]
    max_dx, max_dy = env.action_space.high[:2]
    wall_state_dict = create_walls_from_world_boundaries(
        world_min_x,
        world_max_x,
        world_min_y,
        world_max_y,
        min_dx,
        max_dx,
        min_dy,
        max_dy,
    )
    init_state_dict.update(wall_state_dict)
    init_state = create_state_from_dict(init_state_dict)
    obs, _ = env.reset(seed=123, options={"init_state": init_state})
    assert np.isclose(obs.get(robot, "theta"), 0.0)  # sanity check

    # Move all the way to the right. The number is chosen to be gratuitous.
    right_action = np.zeros_like(env.action_space.high)
    right_action[0] = env.action_space.high[0]
    # Also extend the arm while moving, for fun.
    right_action[3] = env.action_space.high[3]
    for _ in range(25):
        obs, _, _, _, _ = env.step(right_action)

    # Assert that we didn't go off screen.
    assert isinstance(obs, State)
    robot = obs.get_objects(CRVRobotType)[0]
    assert obs.get(robot, "x") < world_max_x

    # Move all the way to the left.
    left_action = np.zeros_like(env.action_space.low)
    left_action[0] = env.action_space.low[0]
    for _ in range(25):
        obs, _, _, _, _ = env.step(left_action)

    # Assert that we didn't go off screen.
    assert obs.get(robot, "x") > world_min_x

    # Rotate and move up.
    up_action = np.zeros_like(env.action_space.high)
    up_action[1] = env.action_space.high[1]
    up_action[2] = env.action_space.high[2]
    for _ in range(25):
        obs, _, _, _, _ = env.step(up_action)
        # Stop rotating when we get past midnight.
        if obs.get(robot, "theta") > np.pi / 2:
            up_action[2] = 0.0

    # Assert that we didn't go off screen.
    assert obs.get(robot, "y") < world_max_y

    # Finish.
    env.close()
