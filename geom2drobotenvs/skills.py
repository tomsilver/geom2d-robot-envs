"""Skills that might be useful in certain environments."""

from typing import Dict, Sequence

import numpy as np
from gym.spaces import Space
from relational_structs.spaces import ObjectSequenceSpace
from relational_structs.structs import (
    Action,
    Object,
    OptionMemory,
    ParameterizedOption,
    State,
)

from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.structs import MultiBody2D, SE2Pose
from geom2drobotenvs.utils import (
    crv_pose_plan_to_action_plan,
    run_motion_planning_for_crv_robot,
    state_has_collision,
)


def create_rectangle_vaccum_pick_option(action_space: Space) -> ParameterizedOption:
    """Use motion planning to get to a pre-pick pose.

    Then extend the arm and turn on the vacuum.
    """

    name = "RectangleVacuumPick"
    params_space = ObjectSequenceSpace([CRVRobotType, RectangleType])

    def _policy(state: State, params: Sequence[Object], memory: OptionMemory) -> Action:
        del state, params  # not used
        import ipdb; ipdb.set_trace()

    def _initiable(
        state: State, params: Sequence[Object], memory: OptionMemory
    ) -> bool:
        robot, target = params

        arm_length = state.get(robot, "arm_length")
        target_width = state.get(target, "width")
        target_height = state.get(target, "height")
        target_cx = state.get(target, "x") + target_width / 2
        target_cy = state.get(target, "y") + target_height / 2

        static_object_body_cache: Dict[Object, MultiBody2D] = {}

        # Try approaching the rectangle from each of four sides, while at the
        # farthest possible distance.
        for approach_theta in [-np.pi / 2, 0, np.pi / 2, np.pi]:

            # Determine the target pose.
            if np.isclose(approach_theta % np.pi, 0.0):  # horizontal approach
                pad = target_width
            else:
                pad = target_height
            approach_dist = arm_length + pad
            approach_dx = -approach_dist * np.cos(approach_theta)
            approach_dy = -approach_dist * np.sin(approach_theta)
            approach_x = target_cx + approach_dx
            approach_y = target_cy + approach_dy
            target_pose = SE2Pose(approach_x, approach_y, approach_theta)

            # Run motion planning.
            pose_plan = run_motion_planning_for_crv_robot(
                state,
                robot,
                target_pose,
                action_space,
                static_object_body_cache=static_object_body_cache,
            )
            if pose_plan is None:
                continue

            # Validate the motion plan by extending the arm and seeing if we
            # would be in collision when the arm is extended.
            target_state = state.copy()
            final_pose = pose_plan[-1]
            target_state.set(robot, "x", final_pose.x)
            target_state.set(robot, "y", final_pose.y)
            target_state.set(robot, "theta", final_pose.theta)
            target_state.set(robot, "arm_joint", arm_length)
            if state_has_collision(target_state, static_object_body_cache):
                continue

            # Found a valid plan; convert it to an action plan and finish.
            action_plan = crv_pose_plan_to_action_plan(pose_plan, action_space)
            memory["plan"] = action_plan
            return True

        # All approach angles failed.
        return False

    def _terminal(state: State, params: Sequence[Object], memory: OptionMemory) -> bool:
        return not memory["plan"]

    return ParameterizedOption(name, params_space, _policy, _initiable, _terminal)
