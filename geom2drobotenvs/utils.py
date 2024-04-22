"""Utilities."""

from typing import Dict, List

import numpy as np
from gym.spaces import Box
from relational_structs.structs import Object, State
from tomsgeoms2d.structs import Circle, Rectangle
from tomsgeoms2d.utils import geom2ds_intersect

from geom2drobotenvs.object_types import CRVRobotType, Geom2DType, RectangleType
from geom2drobotenvs.structs import Body2D, ZOrder, z_orders_may_collide


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
        min_dtheta: float = -np.pi/16,
        max_dtheta: float = np.pi/16,
        min_darm: float = -1e-1,
        max_darm: float = 1e-1,
        min_vac: float = 0.0,
        max_vac: float = 1.0,
    ) -> None:
        low = np.array([min_dx, min_dy, min_dtheta, min_darm, min_vac])
        high = np.array([max_dx, max_dy, max_dtheta, max_darm, max_vac])
        return super().__init__(low, high)


def object_to_body2d(
    obj: Object, state: State, static_object_cache: Dict[Object, Body2D]
) -> Body2D:
    """Create a Body2D instance for objects of standard geom types."""
    if obj.is_instance(CRVRobotType):
        return _robot_to_body2d(obj, state)
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
        geoms = [Rectangle(x, y, width, height, theta)]
        z_orders = [state.get(obj, "z_order")]
        rendering_kwargs = [{
            "facecolor": (state.get(obj, "color_r"), state.get(obj, "color_g"), state.get(obj, "color_b")),
            "edgecolor": "black"
        }]
        body = Body2D(geoms, z_orders, rendering_kwargs)
    else:
        raise NotImplementedError
    if is_static:
        static_object_cache[obj] = body
    return body


def _robot_to_body2d(obj: Object, state: State) -> Body2D:
    """Helper for object_to_body2d()."""
    base = Circle(
        x=state.get(obj, "x"),
        y=state.get(obj, "y"),
        radius=state.get(obj, "base_radius"),
    )
    theta = state.get(obj, "theta")
    arm_joint = state.get(obj, "arm_joint")
    gripper_cx = base.x + np.cos(theta) * arm_joint
    gripper_cy = base.y + np.sin(theta) * arm_joint
    gripper = Rectangle.from_center(
        center_x=gripper_cx,
        center_y=gripper_cy,
        height=(4 * base.radius / 3),
        width=(0.25 * base.radius),
        rotation_about_center=theta,
    )
    link = Rectangle.from_center(
        center_x=(base.x + gripper_cx) / 2,
        center_y=(base.y + gripper_cy) / 2,
        height=np.sqrt((base.x - gripper_cx) ** 2 + (base.y - gripper_cy) ** 2),
        width=(0.5 * gripper.width),
        rotation_about_center=(theta + np.pi / 2),
    )
    geoms = [link, base, gripper]
    z_orders = [ZOrder.SURFACE, ZOrder.ALL, ZOrder.SURFACE]
    silver = (128 / 255, 128 / 255, 128 / 255)
    purple = (128 / 255, 0 / 255, 128 / 255)
    rendering_kwargs = [
        {"facecolor": silver, "edgecolor": "black"},
        {"facecolor": purple, "edgecolor": "black"},
        {"facecolor": purple, "edgecolor": "black"},
    ]
    return Body2D(geoms, z_orders, rendering_kwargs)


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


def state_has_collision(
    state: State, static_object_cache: Dict[Object, Body2D]
) -> bool:
    """Check if a robot or held object has a collision with another object."""
    # TODO handle held objects.
    obj_to_body = {o: object_to_body2d(o, state, static_object_cache) for o in state}
    for robot in state.get_objects(CRVRobotType):
        obstacles = [o for o in state if o != robot]
        robot_body = obj_to_body[robot]
        for robot_geom, robot_z in zip(robot_body.geoms, robot_body.z_orders):
            for obstacle in obstacles:
                obstacle_body = obj_to_body[obstacle]
                for obstacle_geom, obstacle_z in zip(obstacle_body.geoms, obstacle_body.z_orders):
                    if not z_orders_may_collide(robot_z, obstacle_z):
                        continue
                    if geom2ds_intersect(robot_geom, obstacle_geom):
                        return True
    return False
