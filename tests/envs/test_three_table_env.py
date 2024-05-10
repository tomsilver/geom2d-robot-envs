"""Tests for three_table_env.py."""

from typing import Dict, Tuple

import gym
import numpy as np
from relational_structs import Object, ObjectCentricState, ObjectCentricStateSpace
from relational_structs.utils import create_state_from_dict

from geom2drobotenvs.envs.three_table_env import ThreeTableEnv
from geom2drobotenvs.object_types import (
    CRVRobotType,
    Geom2DRobotEnvTypeFeatures,
    RectangleType,
)
from geom2drobotenvs.structs import ZOrder
from geom2drobotenvs.utils import (
    CRVRobotActionSpace,
    create_walls_from_world_boundaries,
    get_tool_tip_position,
    object_to_multibody2d,
)


def test_three_table_env():
    """Basic tests for ThreeTableEnv()."""
    env = ThreeTableEnv()
    assert isinstance(env.observation_space, ObjectCentricStateSpace)
    assert env.observation_space.types == {CRVRobotType, RectangleType}
    assert env.action_space.shape == (5,)
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, ObjectCentricState)


def _get_world_boundaries(env: ThreeTableEnv) -> Tuple[float, float, float, float]:
    # pylint: disable=protected-access
    return (env._world_min_x, env._world_min_y, env._world_max_x, env._world_max_y)


def _create_common_state_dict(env: ThreeTableEnv) -> Dict[Object, Dict[str, float]]:
    """Helper function to create a centered robot and walls."""
    world_min_x, world_min_y, world_max_x, world_max_y = _get_world_boundaries(env)
    # Set up an initial scene with just the robot.
    robot = CRVRobotType("robot")
    init_state_dict = {
        robot: {
            "x": (world_min_x + world_max_x) / 2,
            "y": (world_min_y + world_max_y) / 2,
            "theta": 0.0,
            "base_radius": 0.5,
            "arm_joint": 0.5,
            "arm_length": 3.0,
            "vacuum": 0.0,
            "gripper_height": 0.7,
            "gripper_width": 0.1,
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
    return init_state_dict


def test_three_table_robot_moves():
    """Test basic movements of the robot in ThreeTableEnv()."""
    env = gym.make("geom2drobotenvs/ThreeTables-v0", num_blocks=5)

    # Uncomment to record videos.
    # from gym.wrappers.record_video import RecordVideo
    # env = RecordVideo(env, "unit_test_videos")

    world_min_x, _, world_max_x, world_max_y = _get_world_boundaries(env.unwrapped)

    # Reset the state.
    init_state_dict = _create_common_state_dict(env.unwrapped)
    init_state = create_state_from_dict(init_state_dict, Geom2DRobotEnvTypeFeatures)
    obs, _ = env.reset(seed=123, options={"init_state": init_state})
    assert isinstance(obs, ObjectCentricState)
    robot = obs.get_objects(CRVRobotType)[0]
    assert np.isclose(obs.get(robot, "theta"), 0.0)  # sanity check

    # Move all the way to the right. The number is chosen to be gratuitous.
    right_action = np.zeros_like(env.action_space.high)
    right_action[0] = env.action_space.high[0]
    # Also extend the arm while moving, for fun.
    right_action[3] = env.action_space.high[3]
    for _ in range(25):
        obs, _, _, _, _ = env.step(right_action)

    # Assert that we didn't go off screen.
    assert isinstance(obs, ObjectCentricState)
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


def test_three_table_robot_table_collisions():
    """Test that only the robot base collides with a table."""
    env = ThreeTableEnv()

    # Uncomment to record videos.
    # from gym.wrappers.record_video import RecordVideo
    # env = RecordVideo(env, "unit_test_videos")

    # Reset the state.
    init_state_dict = _create_common_state_dict(env.unwrapped)

    # Add a table to the right of the robot.
    world_min_x, world_min_y, world_max_x, world_max_y = _get_world_boundaries(
        env.unwrapped
    )
    table = RectangleType("table")
    table_width = (world_max_x - world_min_x) / 100.0
    table_height = (world_max_y - world_min_y) / 3.0
    table_right_pad = table_width / 2
    table_x = world_max_x - (10 * table_width + table_right_pad)
    init_state_dict[table] = {
        # Origin is bottom left hand corner.
        "x": table_x,
        "y": (world_min_y + world_max_y - table_height) / 2.0,
        "width": table_width,
        "height": table_height,
        "theta": 0.0,
        "static": True,  # table can't move
        "color_r": 139 / 255,  # brown
        "color_g": 39 / 255,
        "color_b": 19 / 255,
        "z_order": ZOrder.FLOOR.value,
    }

    init_state = create_state_from_dict(init_state_dict, Geom2DRobotEnvTypeFeatures)
    obs, _ = env.reset(seed=123, options={"init_state": init_state})

    # First extend the arm all the way out.
    arm_action = np.zeros_like(env.action_space.high)
    arm_action[3] = env.action_space.high[3]
    for _ in range(25):
        obs, _, _, _, _ = env.step(arm_action)

    # Move all the way to the right.
    right_action = np.zeros_like(env.action_space.high)
    right_action[0] = env.action_space.high[0]
    for _ in range(25):
        obs, _, _, _, _ = env.step(right_action)

    # The robot base should be to the left of the table, but the robot gripper
    # should be to the right of it.
    assert isinstance(obs, ObjectCentricState)
    robot = obs.get_objects(CRVRobotType)[0]
    multibody = object_to_multibody2d(robot, obs, {})
    base = multibody.get_body("base").geom
    gripper = multibody.get_body("gripper").geom
    assert base.x < table_x
    assert gripper.x > table_x

    # Finish.
    env.close()


def test_three_table_vacuum():
    """Tests for picking/placing up one or more objects with the vacuum."""
    env = ThreeTableEnv()

    # Uncomment to record videos.
    # from gym.wrappers.record_video import RecordVideo
    # env = RecordVideo(env, "unit_test_videos")

    assert isinstance(env.action_space, CRVRobotActionSpace)

    # Reset the state.
    init_state_dict = _create_common_state_dict(env.unwrapped)

    # Add a table to the right of the robot.
    world_min_x, world_min_y, world_max_x, world_max_y = _get_world_boundaries(
        env.unwrapped
    )
    table = RectangleType("table")
    table_width = (world_max_x - world_min_x) / 10.0
    table_height = (world_max_y - world_min_y) / 3.0
    table_right_pad = table_width / 2
    table_x = world_max_x - (table_width + table_right_pad)
    table_y = (world_min_y + world_max_y - table_height) / 2.0
    init_state_dict[table] = {
        # Origin is bottom left hand corner.
        "x": table_x,
        "y": table_y,
        "width": table_width,
        "height": table_height,
        "theta": 0.0,
        "static": True,  # table can't move
        "color_r": 139 / 255,  # brown
        "color_g": 39 / 255,
        "color_b": 19 / 255,
        "z_order": ZOrder.FLOOR.value,
    }

    # Add a block on the table.
    block = RectangleType("block")
    block_width = table_width / 5
    block_height = table_height / 3
    block_x = table_x + (table_width - block_width) / 2
    block_y = table_y + (table_height - block_height) / 2
    init_state_dict[block] = {
        "x": block_x,
        "y": block_y,
        "width": block_width,
        "height": block_height,
        "theta": 0.0,
        "static": False,  # block CAN move
        "color_r": 173 / 255,  # blue
        "color_g": 216 / 255,
        "color_b": 230 / 255,
        "z_order": ZOrder.SURFACE.value,  # block is on the table surface
    }

    init_state = create_state_from_dict(init_state_dict, Geom2DRobotEnvTypeFeatures)
    obs, _ = env.reset(seed=123, options={"init_state": init_state})

    # First extend the arm all the way out and turn on the vacuum.
    arm_action = np.zeros_like(env.action_space.high)
    arm_action[3] = env.action_space.high[3]
    arm_action[4] = 1.0
    for _ in range(25):  # gratuitous
        obs, _, _, _, _ = env.step(arm_action)

    # Move to the object that we want to reach.
    assert isinstance(obs, ObjectCentricState)
    robot = obs.get_objects(CRVRobotType)[0]
    block_x = obs.get(block, "x")
    max_dx = env.action_space.high[0]
    for _ in range(25):  # gratuitous
        gripper_x = get_tool_tip_position(obs, robot)[0]
        dx = min(block_x - gripper_x - 1e-6, max_dx)
        if abs(dx) < 1e-5:
            break
        action = np.zeros_like(env.action_space.high)
        action[0] = dx
        action[4] = 1.0  # turn on vacuum
        obs, _, _, _, _ = env.step(action)
    else:
        assert False, "Did not reach object to grasp"

    # Move backward and verify that the block has moved with us.
    left_action = np.zeros_like(env.action_space.high)
    left_action[0] = env.action_space.low[0]
    left_action[4] = 1.0  # turn on vacuum
    for _ in range(5):
        obs, _, _, _, _ = env.step(left_action)

    block_x = obs.get(block, "x")
    assert block_x < table_x

    # Spin around to make sure visually that the block goes with the arm.
    spin_action = np.zeros_like(env.action_space.high)
    spin_action[2] = env.action_space.high[2]
    spin_action[4] = 1.0
    for _ in range(int(2 * np.pi / env.action_space.high[2]) + 1):
        obs, _, _, _, _ = env.step(spin_action)

    block_x = obs.get(block, "x")
    assert block_x < table_x

    # Move forward and put the block back on the table.
    right_action = np.zeros_like(env.action_space.high)
    right_action[0] = env.action_space.high[0]
    right_action[4] = 1.0  # turn on vacuum
    for _ in range(5):
        obs, _, _, _, _ = env.step(right_action)

    # Need to take a noop action to turn off the vacuum.
    env.step(np.zeros_like(env.action_space.high))

    # Move backward without the vacuum on.
    left_action_no_vac = left_action.copy()
    left_action_no_vac[4] = 0.0
    for _ in range(5):
        obs, _, _, _, _ = env.step(left_action_no_vac)

    block_x = obs.get(block, "x")
    assert block_x > table_x

    # Finish.
    env.close()
