"""Tests for skills.py."""

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
    assert option.initiable(obs)
    import ipdb

    ipdb.set_trace()
