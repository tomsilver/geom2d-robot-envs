"""Utilities."""

from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
from gym.spaces import Box
from relational_structs.structs import Object, State, Array
from tomsgeoms2d.structs import Circle, Rectangle
from tomsgeoms2d.utils import geom2ds_intersect
from tomsutils.motion_planning import BiRRT
from tomsutils.utils import get_signed_angle_distance, fig2data

import matplotlib.pyplot as plt
from numpy.typing import NDArray

from geom2drobotenvs.object_types import CRVRobotType, Geom2DType, RectangleType
from geom2drobotenvs.structs import (
    Body2D,
    MultiBody2D,
    SE2Pose,
    ZOrder,
    z_orders_may_collide,
)


class CRVRobotActionSpace(Box):
    """An action space for a CRV robot.

    Actions are bounded relative movements of the base and the arm, as
    well as an absolute setting for the vacuum.
    """

    def __init__(
        self,
        min_dx: float = -5e-1,
        max_dx: float = 5e-1,
        min_dy: float = -5e-1,
        max_dy: float = 5e-1,
        min_dtheta: float = -np.pi / 16,
        max_dtheta: float = np.pi / 16,
        min_darm: float = -1e-1,
        max_darm: float = 1e-1,
        min_vac: float = 0.0,
        max_vac: float = 1.0,
    ) -> None:
        low = np.array([min_dx, min_dy, min_dtheta, min_darm, min_vac])
        high = np.array([max_dx, max_dy, max_dtheta, max_darm, max_vac])
        super().__init__(low, high)


def object_to_multibody2d(
    obj: Object, state: State, static_object_cache: Dict[Object, MultiBody2D]
) -> MultiBody2D:
    """Create a Body2D instance for objects of standard geom types."""
    if obj.is_instance(CRVRobotType):
        return _robot_to_multibody2d(obj, state)
    assert obj.is_instance(Geom2DType)
    is_static = state.get(obj, "static") > 0.5
    if is_static and obj in static_object_cache:
        return static_object_cache[obj]
    if obj.is_instance(RectangleType):
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        width = state.get(obj, "width")
        height = state.get(obj, "height")
        theta = state.get(obj, "theta")
        geom = Rectangle(x, y, width, height, theta)
        z_order = ZOrder(int(state.get(obj, "z_order")))
        rendering_kwargs = {
            "facecolor": (
                state.get(obj, "color_r"),
                state.get(obj, "color_g"),
                state.get(obj, "color_b"),
            ),
            "edgecolor": "black",
        }
        body = Body2D(geom, z_order, rendering_kwargs)
        multibody = MultiBody2D(obj.name, [body])
    else:
        raise NotImplementedError
    if is_static:
        static_object_cache[obj] = multibody
    return multibody


def _robot_to_multibody2d(obj: Object, state: State) -> MultiBody2D:
    """Helper for object_to_multibody2d()."""
    assert obj.is_instance(CRVRobotType)
    bodies: List[Body2D] = []

    # Base.
    base_x = state.get(obj, "x")
    base_y = state.get(obj, "y")
    base_radius = state.get(obj, "base_radius")
    geom = Circle(
        x=base_x,
        y=base_y,
        radius=base_radius,
    )
    z_order = ZOrder.ALL
    purple = (128 / 255, 0 / 255, 128 / 255)
    rendering_kwargs = {"facecolor": purple, "edgecolor": "black"}
    base = Body2D(geom, z_order, rendering_kwargs, name="base")
    bodies.append(base)

    # Gripper.
    theta = state.get(obj, "theta")
    arm_joint = state.get(obj, "arm_joint")
    gripper_cx = base_x + np.cos(theta) * arm_joint
    gripper_cy = base_y + np.sin(theta) * arm_joint
    gripper_height = 4 * base_radius / 3
    gripper_width = 0.25 * base_radius
    geom = Rectangle.from_center(
        center_x=gripper_cx,
        center_y=gripper_cy,
        height=gripper_height,
        width=gripper_width,
        rotation_about_center=theta,
    )
    z_order = ZOrder.SURFACE
    rendering_kwargs = {"facecolor": purple, "edgecolor": "black"}
    gripper = Body2D(geom, z_order, rendering_kwargs, name="gripper")
    bodies.append(gripper)

    # Arm.
    geom = Rectangle.from_center(
        center_x=(base_x + gripper_cx) / 2,
        center_y=(base_y + gripper_cy) / 2,
        height=np.sqrt((base_x - gripper_cx) ** 2 + (base_y - gripper_cy) ** 2),
        width=(0.5 * gripper_width),
        rotation_about_center=(theta + np.pi / 2),
    )
    z_order = ZOrder.SURFACE
    silver = (128 / 255, 128 / 255, 128 / 255)
    rendering_kwargs = {"facecolor": silver, "edgecolor": "black"}
    arm = Body2D(geom, z_order, rendering_kwargs, name="arm")
    bodies.append(arm)

    # If the vacuum is on, add a suction area.
    if state.get(obj, "vacuum") > 0.5:
        suction_height = gripper_height
        suction_width = gripper_width / 3
        suction_cx = base_x + np.cos(theta) * (arm_joint + gripper_width / 2)
        suction_cy = base_y + np.sin(theta) * (arm_joint + gripper_width / 2)
        geom = Rectangle.from_center(
            center_x=suction_cx,
            center_y=suction_cy,
            height=suction_height,
            width=suction_width,
            rotation_about_center=theta,
        )
        z_order = ZOrder.NONE  # NOTE: suction collides with nothing
        yellow = (255 / 255, 255 / 255, 153 / 255)
        rendering_kwargs = {"facecolor": yellow}
        suction = Body2D(geom, z_order, rendering_kwargs, name="suction")
        bodies.append(suction)

    return MultiBody2D(obj.name, bodies)


def create_walls_from_world_boundaries(
    world_min_x: float,
    world_max_x: float,
    world_min_y: float,
    world_max_y: float,
    min_dx: float,
    max_dx: float,
    min_dy: float,
    max_dy: float,
) -> Dict[Object, Dict[str, float]]:
    """Create wall objects and feature dicts based on world boundaries.

    Velocities are used to determine how large the walls need to be to
    avoid the possibility that the robot will transport over the wall.
    """
    state_dict: Dict[Object, Dict[str, float]] = {}
    # Right wall.
    right_wall = RectangleType("right_wall")
    side_wall_height = world_max_y - world_min_y
    state_dict[right_wall] = {
        "x": world_max_x,
        "y": world_min_y,
        "width": 2 * max_dx,  # 2x just for safety
        "height": side_wall_height,
        "theta": 0.0,
        "static": True,
        "color_r": 0.1,
        "color_g": 0.1,
        "color_b": 0.1,
        "z_order": ZOrder.ALL.value,
    }
    # Left wall.
    left_wall = RectangleType("left_wall")
    state_dict[left_wall] = {
        "x": world_min_x + 2 * min_dx,
        "y": world_min_y,
        "width": 2 * abs(min_dx),  # 2x just for safety
        "height": side_wall_height,
        "theta": 0.0,
        "static": True,
        "color_r": 0.1,
        "color_g": 0.1,
        "color_b": 0.1,
        "z_order": ZOrder.ALL.value,
    }
    # Top wall.
    top_wall = RectangleType("top_wall")
    horiz_wall_width = 2 * 2 * abs(min_dx) + world_max_x - world_min_x
    state_dict[top_wall] = {
        "x": world_min_x + 2 * min_dx,
        "y": world_max_y,
        "width": horiz_wall_width,
        "height": 2 * max_dy,
        "theta": 0.0,
        "static": True,
        "color_r": 0.1,
        "color_g": 0.1,
        "color_b": 0.1,
        "z_order": ZOrder.ALL.value,
    }
    # Bottom wall.
    bottom_wall = RectangleType("bottom_wall")
    state_dict[bottom_wall] = {
        "x": world_min_x + 2 * min_dx,
        "y": world_min_y + 2 * min_dy,
        "width": horiz_wall_width,
        "height": 2 * max_dy,
        "theta": 0.0,
        "static": True,
        "color_r": 0.1,
        "color_g": 0.1,
        "color_b": 0.1,
        "z_order": ZOrder.ALL.value,
    }
    return state_dict


def render_state(state: State, static_object_body_cache: Optional[Dict[Object, MultiBody2D]] = None,
                 world_min_x: float= 0.0, world_max_x: float = 10.0,
                 world_min_y: float=0.0, world_max_y: float = 10.0,
                 render_dpi: int = 150) -> NDArray[np.uint8]:
    """Render a state. Useful for viz and debugging."""
    figsize = (
        world_max_x - world_min_x,
        world_max_y - world_min_y,
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=render_dpi)

    # Sort objects by ascending z order, with the robot first.
    def _render_order(obj: Object) -> int:
        if obj.is_instance(CRVRobotType):
            return -1
        return int(state.get(obj, "z_order"))

    for obj in sorted(state, key=_render_order):
        body = object_to_multibody2d(obj, state, static_object_body_cache)
        body.plot(ax)

    pad_x = (world_max_x - world_min_x) / 25
    pad_y = (world_max_y - world_min_y) / 25
    ax.set_xlim(world_min_x - pad_x, world_max_x + pad_x)
    ax.set_ylim(world_min_y - pad_y, world_max_y + pad_y)
    ax.axis("off")
    plt.tight_layout()
    img = fig2data(fig)
    plt.clf()
    return img


def state_has_collision(
    state: State, static_object_cache: Dict[Object, MultiBody2D]
) -> bool:
    """Check if a robot or held object has a collision with another object."""
    # NOTE: need to handle held objects.
    obj_to_multibody = {
        o: object_to_multibody2d(o, state, static_object_cache) for o in state
    }
    for robot in state.get_objects(CRVRobotType):
        obstacles = [o for o in state if o != robot]
        robot_multibody = obj_to_multibody[robot]
        for robot_body in robot_multibody.bodies:
            for obstacle in obstacles:
                obstacle_multibody = obj_to_multibody[obstacle]
                for obstacle_body in obstacle_multibody.bodies:
                    if not z_orders_may_collide(
                        robot_body.z_order, obstacle_body.z_order
                    ):
                        continue
                    if geom2ds_intersect(robot_body.geom, obstacle_body.geom):
                        return True
    return False


def get_tool_tip_position(state: State, robot: Object) -> Tuple[float, float]:
    """Get the tip of the tool for the robot, which is defined as the center of
    the bottom edge of the gripper."""
    multibody = _robot_to_multibody2d(robot, state)
    gripper_geom = multibody.get_body("gripper").geom
    assert isinstance(gripper_geom, Rectangle)
    # Transform the x, y point.
    tool_tip = np.array([1.0, 0.5])
    scale_matrix = np.array(
        [
            [gripper_geom.width, 0],
            [0, gripper_geom.height],
        ]
    )
    translate_vector = np.array([gripper_geom.x, gripper_geom.y])
    tool_tip = tool_tip @ scale_matrix.T
    tool_tip = tool_tip @ gripper_geom.rotation_matrix.T
    tool_tip = translate_vector + tool_tip
    return (tool_tip[0], tool_tip[1])


def get_se2_pose(state: State, obj: Object) -> SE2Pose:
    """Get the SE2Pose of an object in a given state."""
    return SE2Pose(
        x=state.get(obj, "x"),
        y=state.get(obj, "y"),
        theta=state.get(obj, "theta"),
    )


def get_relative_se2_transform(state: State, obj1: Object, obj2: Object) -> SE2Pose:
    """Get the pose of obj2 in the frame of obj1."""
    world_to_obj1 = get_se2_pose(state, obj1)
    world_to_obj2 = get_se2_pose(state, obj2)
    return world_to_obj1.inverse * world_to_obj2


def get_suctioned_objects(state: State, robot: Object) -> List[Tuple[Object, SE2Pose]]:
    """Find objects that are in the suction zone of a CRVRobot and return the
    associated transform from robot to suctioned object."""
    # If the robot's vacuum is not on, there are no suctioned objects.
    if state.get(robot, "vacuum") <= 0.5:
        return []
    robot_multibody = _robot_to_multibody2d(robot, state)
    suction_body = robot_multibody.get_body("suction")
    # Find MOVABLE objects in collision with the suction geom.
    movable_objects = [o for o in state if o != robot and state.get(o, "static") < 0.5]
    suctioned_objects: List[Object] = []
    for obj in movable_objects:
        # No point in using a static object cache because these objects are
        # not static by definition.
        obj_multibody = object_to_multibody2d(obj, state, {})
        for obj_body in obj_multibody.bodies:
            if geom2ds_intersect(suction_body.geom, obj_body.geom):
                robot_to_obj = get_relative_se2_transform(state, robot, obj)
                suctioned_objects.append((obj, robot_to_obj))
    return suctioned_objects


def run_motion_planning_for_crv_robot(state: State, robot: Object, target_pose: SE2Pose, action_space: CRVRobotActionSpace,
                                      vacuum_while_moving: bool = False,
                                      seed: int = 0, num_attempts: int = 10, num_iters: int = 100, smooth_amt: int = 50
                                      ) -> Optional[List[Array]]:
    """Run motion planning in an environment with a CRV action space."""

    rng = np.random.default_rng(seed)
    
    # Use the object positions in the state to create a rough room boundary.
    x_lb, x_ub, y_lb, y_ub = np.inf, -np.inf, np.inf, -np.inf
    for obj in state:
        pose = get_se2_pose(state, obj)
        x_lb = min(x_lb, pose.x)
        x_ub = max(x_ub, pose.x)
        y_lb = min(y_lb, pose.y)
        y_ub = max(y_ub, pose.y)

    # Create a static version of the state so that the geoms only need to be
    # instantiated once during motion planning (except for the robot).
    static_state = state.copy()
    for o in static_state:
        if o.is_instance(CRVRobotType):
            continue
        static_state.set(o, "static", 1.0)
    static_object_body_cache: Dict[Object, MultiBody2D] = {}

    # Set up the RRT methods.
    def sample_fn(_: SE2Pose) -> SE2Pose:
        """Sample a robot pose."""
        x = rng.uniform(x_lb, x_ub)
        y = rng.uniform(y_lb, y_ub)
        theta = rng.uniform(-np.pi, np.pi)
        return SE2Pose(x, y, theta)
    
    def extend_fn(pt1: SE2Pose, pt2: SE2Pose) -> Iterable[SE2Pose]:
        """Interpolate between the two poses."""
        # Make sure that we obey the bounds on actions.
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dtheta = get_signed_angle_distance(pt2.theta, pt1.theta)
        assert isinstance(action_space, CRVRobotActionSpace)
        abs_x = action_space.high[0] if dx > 0 else action_space.low[0]
        abs_y = action_space.high[1] if dy > 0 else action_space.low[1]
        abs_theta = action_space.high[2] if dtheta > 0 else action_space.low[2]
        x_num_steps = dx / abs_x
        assert x_num_steps >= 0
        y_num_steps = dy / abs_y
        assert y_num_steps >= 0
        theta_num_steps = dtheta / abs_theta
        assert theta_num_steps >= 0
        num_steps = max(x_num_steps, y_num_steps, theta_num_steps)
        x_interp = np.linspace(pt1.x, pt2.x, num=num_steps, endpoint=True)
        y_interp = np.linspace(pt1.y, pt2.y, num=num_steps, endpoint=True)
        theta_interp = np.linspace(pt1.theta, pt2.theta, num=num_steps, endpoint=True)
        for x, y, theta in zip(x_interp, y_interp, theta_interp):
            yield SE2Pose(x, y, theta)

    def collision_fn(pt: SE2Pose) -> bool:
        """Check for collisions if the robot were at this pose."""
        # Update the static state with the robot's new hypothetical pose.
        static_state.set(robot, "x", pt.x)
        static_state.set(robot, "y", pt.y)
        static_state.set(robot, "theta", pt.theta)
        return state_has_collision(static_state, static_object_body_cache)
    
    def distance_fn(pt1: SE2Pose, pt2: SE2Pose) -> float:
        """Return a distance between the two points."""
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dtheta = get_signed_angle_distance(pt2.theta, pt1.theta)
        return np.sqrt(dx**2 + dy**2) + abs(dtheta)

    birrt = BiRRT(sample_fn, extend_fn, collision_fn, distance_fn, rng, num_attempts,
                num_iters, smooth_amt)
    
    initial_pose = get_se2_pose(state, robot)
    pose_plan = birrt.query(initial_pose, target_pose)
    if pose_plan is None:
        return None
    
    action_plan: List[Array] = []
    for pt1, pt2 in zip(pose_plan[:-1], pose_plan[1:]):
        action = np.zeros_like(action_space.high)
        action[0] = pt2.x - pt1.x
        action[1] = pt2.y - pt1.y
        action[2] = pt2.theta - pt1.theta
        action[4] = 1.0 if vacuum_while_moving else 0.0
        action_plan.append(action)

    return action_plan