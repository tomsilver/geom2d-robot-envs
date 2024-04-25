"""Tests for skills.py."""

import numpy as np
from relational_structs.structs import State

from geom2drobotenvs.envs.three_table_env import ThreeTableEnv
from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.skills import create_rectangle_vaccum_pick_option


def test_create_rectangle_vaccum_pick_option():
    """Tests for create_rectangle_vaccum_pick_option()."""
    env = ThreeTableEnv()

    parameterized_opt = create_rectangle_vaccum_pick_option(env.action_space)
    obs, _ = env.reset()
    assert isinstance(obs, State)

    # Try to pick up the smallest block.
    robot = obs.get_objects(CRVRobotType)[0]
    blocks = [
        o for o in obs if o.is_instance(RectangleType) and obs.get(o, "static") < 0.5
    ]
    block = min(blocks, key=lambda b: obs.get(b, "width") * obs.get(b, "height"))
    option = parameterized_opt.ground([robot, block])

    # Move the robot to a more interesting initial location.
    obs.set(robot, "x", obs.get(robot, "x") - 3.0)
    obs.set(robot, "y", obs.get(robot, "y") - 3.0)
    obs.set(robot, "theta", 2 * np.pi / 3)
    obs, _ = env.reset(options={"init_state": obs})

    # Uncomment to record videos.
    # from gym.wrappers.record_video import RecordVideo
    # env = RecordVideo(env, "unit_test_videos")

    assert option.initiable(obs)
    for _ in range(100):  # gratuitous
        act = option.policy(obs)
        obs, _, _, _, _ = env.step(act)
        if option.terminal(obs):
            break
    else:
        assert False, "Option did not terminate."

    env.close()
