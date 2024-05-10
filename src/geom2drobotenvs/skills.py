"""Skills that might be useful in certain environments."""

from typing import Dict, Iterator, List, Sequence

import numpy as np
from gymnasium.spaces import Space
from relational_structs import (
    Action,
    Object,
    ObjectCentricState,
    ObjectSequenceSpace,
    OptionMemory,
    ParameterizedOption,
)

from geom2drobotenvs.concepts import is_inside
from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.structs import MultiBody2D, SE2Pose
from geom2drobotenvs.utils import (
    CRVRobotActionSpace,
    crv_pose_plan_to_action_plan,
    get_suctioned_objects,
    rectangle_object_to_geom,
    run_motion_planning_for_crv_robot,
    snap_suctioned_objects,
    state_has_collision,
)


def _iter_motion_plans_to_rectangle(
    state: ObjectCentricState,
    robot: Object,
    target: Object,
    action_space: Space,
    robot_to_target_side_dist: float,
    static_object_body_cache: Dict[Object, MultiBody2D],
) -> Iterator[List[SE2Pose]]:
    """Helper for picking and placing that generates motion plans to approach a
    rectangle from four possible sides."""

    target_rect = rectangle_object_to_geom(state, target, static_object_body_cache)
    target_width = target_rect.width
    target_height = target_rect.height
    target_cx = target_rect.center[0]
    target_cy = target_rect.center[1]
    target_theta = target_rect.theta
    world_to_target = SE2Pose(target_cx, target_cy, target_theta)

    # Try approaching the rectangle from each of four sides, while at the
    # farthest possible distance.
    for approach_theta, target_size in [
        (-np.pi / 2, target_height),
        (0, target_width),
        (np.pi / 2, target_height),
        (np.pi, target_width),
    ]:
        # Determine the approach pose relative to target.
        target_pad = target_size / 2
        approach_dist = robot_to_target_side_dist + target_pad
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
        if pose_plan is not None:
            yield pose_plan


def create_rectangle_vaccum_pick_option(action_space: Space) -> ParameterizedOption:
    """Use motion planning to get to a pre-pick pose, extend the arm, turn on
    the vacuum, and then retract the arm."""

    name = "RectangleVacuumPick"
    params_space = ObjectSequenceSpace([CRVRobotType, RectangleType])
    assert isinstance(action_space, CRVRobotActionSpace)

    def _policy(
        state: ObjectCentricState, params: Sequence[Object], memory: OptionMemory
    ) -> Action:
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
        state: ObjectCentricState, params: Sequence[Object], memory: OptionMemory
    ) -> bool:
        robot, target = params

        # Try approaching the rectangle from each of four sides, while at the
        # farthest possible distance.
        arm_length = state.get(robot, "arm_length")
        gripper_width = state.get(robot, "gripper_width")
        gripper_pad = gripper_width / 2
        vacuum_pad = 1e-6  # leave a small space to avoid collisions
        robot_to_target_side_dist = arm_length + gripper_pad + vacuum_pad
        static_object_body_cache: Dict[Object, MultiBody2D] = {}
        for pose_plan in _iter_motion_plans_to_rectangle(
            state,
            robot,
            target,
            action_space,
            robot_to_target_side_dist,
            static_object_body_cache,
        ):
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

    def _terminal(
        state: ObjectCentricState, params: Sequence[Object], memory: OptionMemory
    ) -> bool:
        robot, target = params
        robot_radius = state.get(robot, "base_radius")
        arm_joint = state.get(robot, "arm_joint")
        arm_retracted = abs(robot_radius - arm_joint) < 1e-5
        grasped_objects = {o for o, _ in get_suctioned_objects(state, robot)}
        target_grasped = target in grasped_objects
        return not memory["move_plan"] and arm_retracted and target_grasped

    return ParameterizedOption(name, params_space, _policy, _initiable, _terminal)


def create_rectangle_vaccum_table_place_option(
    action_space: Space,
) -> ParameterizedOption:
    """Use motion planning to get to a pre-place pose, extend the arm, turn off
    the vacuum, and then retract the arm.

    Considers discrete set of placements starting at the back of the
    table and then moving towards the front until no collisions are
    detected.
    """

    name = "RectangleVacuumPlace"
    # robot, held object, target table
    params_space = ObjectSequenceSpace([CRVRobotType, RectangleType, RectangleType])
    assert isinstance(action_space, CRVRobotActionSpace)

    def _policy(
        state: ObjectCentricState, params: Sequence[Object], memory: OptionMemory
    ) -> Action:
        # This policy is completely open-loop.
        del state, params  # unused
        return memory["action_plan"].pop(0)

    def _initiable(
        state: ObjectCentricState, params: Sequence[Object], memory: OptionMemory
    ) -> bool:
        robot, held_obj, table = params

        static_object_body_cache: Dict[Object, MultiBody2D] = {}
        suctioned_objs = get_suctioned_objects(state, robot)

        # Try to approach the table from each of the four sides. After each
        # motion plan, try to extend the arm as far as possible until a
        # collision is about to occur. If the held object is on the table,
        # return that plan. Otherwise, try the next of the four approaches.
        robot_base_radius = state.get(robot, "base_radius")
        held_obj_thickness = min(
            state.get(held_obj, "width"), state.get(held_obj, "height")
        )
        pad = 0.25 * robot_base_radius
        robot_to_target_side_dist = robot_base_radius + held_obj_thickness + pad
        for pose_plan in _iter_motion_plans_to_rectangle(
            state,
            robot,
            table,
            action_space,
            robot_to_target_side_dist,
            static_object_body_cache,
        ):

            # Simulate extending the arm from the last pose in the plan.
            sim_state = state.copy()
            sim_state.set(robot, "x", pose_plan[-1].x)
            sim_state.set(robot, "y", pose_plan[-1].y)
            sim_state.set(robot, "theta", pose_plan[-1].theta)
            snap_suctioned_objects(sim_state, robot, suctioned_objs)

            num_arm_extensions = 0
            arm_length = sim_state.get(robot, "arm_length")
            arm_joint = sim_state.get(robot, "arm_joint")
            arm_dx = action_space.high[3]
            while arm_joint + 1e-6 < arm_length:
                arm_joint += arm_dx
                sim_state.set(robot, "arm_joint", arm_joint)
                snap_suctioned_objects(sim_state, robot, suctioned_objs)
                if state_has_collision(sim_state, static_object_body_cache):
                    # Roll back the change.
                    arm_joint -= arm_dx
                    sim_state.set(robot, "arm_joint", arm_joint)
                    snap_suctioned_objects(sim_state, robot, suctioned_objs)
                    break
                num_arm_extensions += 1

            # Check if the held object is on the table. If so, finish the plan.
            if is_inside(sim_state, held_obj, table, static_object_body_cache):
                action_plan = crv_pose_plan_to_action_plan(
                    pose_plan, action_space, vacuum_while_moving=True
                )
                # Extend arm.
                arm_extend = np.array(
                    [0.0, 0.0, 0.0, action_space.high[3], action_space.high[4]],
                    dtype=np.float32,
                )
                for _ in range(num_arm_extensions):
                    action_plan.append(arm_extend)
                # Release the vacuum.
                drop_action = np.zeros(5, dtype=action_space.dtype)
                action_plan.append(drop_action)
                # Retract the arm.
                arm_retract = np.array(
                    [0.0, 0.0, 0.0, action_space.low[3], 0.0], dtype=np.float32
                )
                for _ in range(num_arm_extensions):
                    action_plan.append(arm_retract)
                # Store the plan.
                memory["action_plan"] = action_plan
                return True

        # All approach angles failed.
        return False

    def _terminal(
        state: ObjectCentricState, params: Sequence[Object], memory: OptionMemory
    ) -> bool:
        del state, params
        return not memory["action_plan"]

    return ParameterizedOption(name, params_space, _policy, _initiable, _terminal)
