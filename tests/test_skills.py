"""Tests for skills.py."""

import numpy as np
from relational_structs.structs import State

from geom2drobotenvs.envs.three_table_env import ThreeTableEnv
from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.skills import create_rectangle_vaccum_pick_option,create_rectangle_vaccum_table_place_option
from geom2drobotenvs.utils import get_suctioned_objects
from geom2drobotenvs.structs import ZOrder


def test_crv_pick_and_place():
    """Tests for pick and place skills with the CRV robot."""
    env = ThreeTableEnv()

    pick = create_rectangle_vaccum_pick_option(env.action_space)
    place = create_rectangle_vaccum_table_place_option(env.action_space)

    obs, _ = env.reset()
    assert isinstance(obs, State)

    # Try to pick up the smallest block.
    robot = obs.get_objects(CRVRobotType)[0]
    blocks = [
        o for o in obs if o.is_instance(RectangleType) and obs.get(o, "static") < 0.5
    ]
    block = min(blocks, key=lambda b: obs.get(b, "width") * obs.get(b, "height"))
    pick_block = pick.ground([robot, block])

    # Afterward, place the object on the farthest away table.
    tables = [
        o for o in obs if o.is_instance(RectangleType) and \
            obs.get(o, "static") > 0.5 and \
            int(obs.get(o, "z_order")) == ZOrder.FLOOR.value
    ]
    bx = obs.get(block, "x")
    by = obs.get(block, "y")
    dist = lambda t: (bx - obs.get(t, "x"))**2 + (by - obs.get(t, "y"))**2
    table = max(tables, key=dist)
    place_block = place.ground([robot, block, table])

    # Move the robot to a more interesting initial location.
    obs.set(robot, "x", obs.get(robot, "x") - 3.0)
    obs.set(robot, "y", obs.get(robot, "y") - 3.0)
    obs.set(robot, "theta", 2 * np.pi / 3)
    obs, _ = env.reset(options={"init_state": obs})

    # Uncomment to record videos.
    # from gym.wrappers.record_video import RecordVideo
    # env = RecordVideo(env, "unit_test_videos")

    assert pick_block.initiable(obs)
    for _ in range(100):  # gratuitous
        act = pick_block.policy(obs)
        obs, _, _, _, _ = env.step(act)
        if pick_block.terminal(obs):
            break
    else:
        assert False, "Pick option did not terminate."

    suctioned_objs = {o for o, _ in get_suctioned_objects(obs, robot)}
    assert block in suctioned_objs

    # Place the object on the farthest away table.
    assert place_block.initiable(obs)
    for _ in range(100):  # gratuitous
        act = place_block.policy(obs)
        obs, _, _, _, _ = env.step(act)
        if place_block.terminal(obs):
            break
    else:
        assert False, "Place option did not terminate."

    # TODO assert something

    import ipdb; ipdb.set_trace()

    env.close()
