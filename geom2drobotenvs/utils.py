"""Utilities."""

from typing import Dict, List

import numpy as np
from gym.spaces import Box
from relational_structs.structs import Object, State
from tomsgeoms2d.structs import Circle, Geom2D, Rectangle

from geom2drobotenvs.object_types import CRVRobotType, Geom2DType, RectangleType


class CRVRobotActionSpace(Box):
    """An action space for a CRV robot.

    Actions are bounded relative movements of the base and the arm, as
    well as an absolute setting for the vacuum.
    """

    def __init__(
        self,
        min_dx: float = -1e-1,
        max_dx: float = 1e-1,
        min_dy: float = -1e-1,
        max_dy: float = 1e-1,
        min_dtheta: float = -1.0,
        max_dtheta: float = 1.0,
        min_darm: float = -1e-1,
        max_darm: float = 1e-1,
        min_vac: float = 0.0,
        max_vac: float = 1.0,
    ) -> None:
        low = np.array([min_dx, min_dy, min_dtheta, min_darm, min_vac])
        high = np.array([max_dx, max_dy, max_dtheta, max_darm, max_vac])
        return super().__init__(low, high)


def object_to_geom2d_list(
    obj: Object, state: State, static_object_cache: Dict[Object, List[Geom2D]]
) -> List[Geom2D]:
    """Create a list of Geom2D instances for objects of standard geom types."""
    if obj.is_instance(CRVRobotType):
        return _robot_to_geom2d_list(obj, state)
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
    else:
        raise NotImplementedError
    if is_static:
        static_object_cache[obj] = geoms
    return geoms


def _robot_to_geom2d_list(obj: Object, state: State) -> List[Geom2D]:
    """Helper for object_to_geom2d_list()."""
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
    return [link, base, gripper]
