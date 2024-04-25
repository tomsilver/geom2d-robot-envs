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
    CRVRobotActionSpace,
    crv_pose_plan_to_action_plan,
    get_suctioned_objects,
    run_motion_planning_for_crv_robot,
    state_has_collision,
)


def create_rectangle_vaccum_pick_option(action_space: Space) -> ParameterizedOption:
    """Use motion planning to get to a pre-pick pose, extend the arm, turn on
    the vacuum, and then retract the arm."""

    name = "RectangleVacuumPick"
    params_space = ObjectSequenceSpace([CRVRobotType, RectangleType])
    assert isinstance(action_space, CRVRobotActionSpace)

    def _policy(state: State, params: Sequence[Object], memory: OptionMemory) -> Action:
        robot, target = params

        # If the target is grasped, retract right away.
        grasped_objects = {o for o, _ in get_suctioned_objects(state, robot)}
        if target in grasped_objects:
            return np.array(
                [0.0, 0.0, 0.0, action_space.low[3], action_space.high[4]],
                dtype=np.float32,
            )

        # Moving is finished.
        if not memory["move_plan"]:
            # Arm is extended, so turn on the vacuum.
            arm_length = state.get(robot, "arm_length")
            arm_joint = state.get(robot, "arm_joint")
            arm_extended = abs(arm_length - arm_joint) < 1e-5
            if arm_extended:
                return np.array(
                    [0.0, 0.0, 0.0, 0.0, action_space.high[4]], dtype=np.float32
                )
            # Extend the arm.
            return np.array(
                [0.0, 0.0, 0.0, action_space.high[3], 0.0], dtype=np.float32
            )
        # Move.
        return memory["move_plan"].pop(0)

    def _initiable(
        state: State, params: Sequence[Object], memory: OptionMemory
    ) -> bool:
        robot, target = params

        arm_length = state.get(robot, "arm_length")
        gripper_width = state.get(robot, "gripper_width")
        target_width = state.get(target, "width")
        target_height = state.get(target, "height")
        target_cx = state.get(target, "x") + target_width / 2
        target_cy = state.get(target, "y") + target_height / 2
        target_theta = state.get(target, "theta")
        world_to_target = SE2Pose(target_cx, target_cy, target_theta)

        static_object_body_cache: Dict[Object, MultiBody2D] = {}

        # Try approaching the rectangle from each of four sides, while at the
        # farthest possible distance.
        for approach_theta in [-np.pi / 2, 0, np.pi / 2, np.pi]:

            # Determine the approach pose relative to target.
            if np.isclose(approach_theta % np.pi, 0.0):  # horizontal approach
                target_pad = target_width / 2
            else:
                target_pad = target_height / 2
            gripper_pad = gripper_width / 2
            vacuum_pad = 1e-6  # leave a small space to avoid collisions
            approach_dist = arm_length + target_pad + gripper_pad + vacuum_pad
            approach_x = -approach_dist * np.cos(approach_theta)
            approach_y = -approach_dist * np.sin(approach_theta)
            target_to_robot = SE2Pose(approach_x, approach_y, approach_theta)
            # Convert to absolute pose.
            target_pose = world_to_target * target_to_robot

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
            memory["move_plan"] = action_plan
            return True

        # All approach angles failed.
        return False

    def _terminal(state: State, params: Sequence[Object], memory: OptionMemory) -> bool:
        robot, target = params
        robot_radius = state.get(robot, "base_radius")
        arm_joint = state.get(robot, "arm_joint")
        arm_retracted = abs(robot_radius - arm_joint) < 1e-5
        grasped_objects = {o for o, _ in get_suctioned_objects(state, robot)}
        target_grasped = target in grasped_objects
        return not memory["move_plan"] and arm_retracted and target_grasped

    return ParameterizedOption(name, params_space, _policy, _initiable, _terminal)
