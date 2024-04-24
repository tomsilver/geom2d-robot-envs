"""Skills that might be useful in certain environments."""

from relational_structs.structs import ParameterizedOption, State, Object, OptionMemory, Action
from relational_structs.spaces import ObjectSequenceSpace
from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.utils import run_motion_planning_for_crv_robot, get_se2_pose
from gym.spaces import Space
from typing import Sequence
import numpy as np


def create_rectangle_vaccum_pick_option(action_space: Space) -> ParameterizedOption:
    """Use motion planning to get to a pre-pick pose. Then extend the arm and
    turn on the vacuum."""

    name = "RectangleVacuumPick"
    params_space = ObjectSequenceSpace([CRVRobotType, RectangleType])

    def _policy(state: State, params: Sequence[Object], memory: OptionMemory) -> Action:
        del state, params  # not used

    def _initiable(state: State, params: Sequence[Object], memory: OptionMemory) -> bool:
        robot, target = params

        # Determine the distance that the robot needs to be from the target in
        # order for the arm to reach it.
        arm_length = state.get(robot, "arm_length")
        import ipdb; ipdb.set_trace()

        # Try approaching the rectangle from each of four sides, while at the
        # farthest possible distance.
        # for approach_direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

        # plan = run_motion_planning_for_crv_robot(state, robot
        # if plan is None:
        #     return False
        # memory["plan"] = plan
        # return True

    def _terminal(state: State, params: Sequence[Object], memory: OptionMemory) -> bool:
        return not memory["plan"]

    return ParameterizedOption(name, params_space, _policy, _initiable, _terminal)
