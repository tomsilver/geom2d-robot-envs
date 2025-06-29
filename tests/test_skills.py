"""Tests for skills.py."""

from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from relational_structs import ObjectCentricState

from geom2drobotenvs.concepts import is_inside
from geom2drobotenvs.envs.three_table_env import ThreeTableEnv
from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.skills import (
    create_rectangle_vaccum_pick_option,
    create_rectangle_vaccum_table_place_inside_option,
)
from geom2drobotenvs.structs import ZOrder


def test_crv_pick_and_place():
    """Tests for pick and place skills with the CRV robot."""
    env = ThreeTableEnv(num_blocks=4)

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    pick = create_rectangle_vaccum_pick_option(env.action_space)
    place = create_rectangle_vaccum_table_place_inside_option(env.action_space)

    obs, _ = env.reset()
    assert isinstance(obs, ObjectCentricState)

    # Get the objects.
    robot = obs.get_objects(CRVRobotType)[0]
    # Order blocks from smallest to largest (area).
    blocks = [
        o for o in obs if o.is_instance(RectangleType) and obs.get(o, "static") < 0.5
    ]
    blocks = sorted(blocks, key=lambda b: obs.get(b, "width") * obs.get(b, "height"))
    # Put the blocks in the farthest away table.
    tables = [
        o
        for o in obs
        if o.is_instance(RectangleType)
        and obs.get(o, "static") > 0.5
        and int(obs.get(o, "z_order")) == ZOrder.FLOOR.value
    ]
    bx = obs.get(blocks[0], "x")
    by = obs.get(blocks[0], "y")
    dist = lambda t: (bx - obs.get(t, "x")) ** 2 + (by - obs.get(t, "y")) ** 2
    table = max(tables, key=dist)

    for block in blocks:
        pick_block = pick.ground([robot, block])
        place_block = place.ground([robot, block, table])
        for option in [pick_block, place_block]:
            assert option.initiable(obs)
            for _ in range(100):  # gratuitous
                act = option.policy(obs)
                obs, _, _, _, _ = env.step(act)
                if option.terminal(obs):
                    break
            else:
                assert False, f"Option {option} did not terminate."

    static_object_cache = {}
    for block in blocks:
        is_inside(obs, block, table, static_object_cache)

    env.close()
