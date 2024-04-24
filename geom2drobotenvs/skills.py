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

        # Try approaching the rectangle from each of four sides, while at the
        # farthest possible distance.
        for approach_theta in [0, np.pi / 2, np.pi, -np.pi / 2]:
            # Determine the target pose.
            arm_length = state.get(robot, "arm_length")
            import ipdb; ipdb.set_trace()

            # TODO: validate the motion plan by extending the arm and seeing if we
            # would be in collision when the arm is extended.

            # plan = run_motion_planning_for_crv_robot(state, robot
            # if plan is None:
            #     return False
            # memory["plan"] = plan
            # return True

    def _terminal(state: State, params: Sequence[Object], memory: OptionMemory) -> bool:
        return not memory["plan"]

    return ParameterizedOption(name, params_space, _policy, _initiable, _terminal)
