"""Object types that are common across different environments."""

from typing import Dict, List

from relational_structs import Type

Geom2DRobotEnvTypeFeatures: Dict[Type, List[str]] = {}

# All geoms have an origin (x, y) and a rotation (in radians), and a bit
# indicating whether the geom is static (versus movable). They also have RGB.
# The z_order is an integer used for collision checking.
Geom2DType = Type("geom2d")
Geom2DRobotEnvTypeFeatures[Geom2DType] = [
    "x",
    "y",
    "theta",
    "static",
    "color_r",
    "color_g",
    "color_b",
    "z_order",
]
# Specific geom types.
RectangleType = Type("rectangle", parent=Geom2DType)
Geom2DRobotEnvTypeFeatures[RectangleType] = Geom2DRobotEnvTypeFeatures[Geom2DType] + [
    "width",
    "height",
]
# A robot with a circle base, a rectangle arm, and a vacuum rectangle gripper.
# The (x, y, theta) are for the center of the robot base circle. The base_radius
# is for that circle. The arm_joint is a distance between the center and the
# gripper. The arm_length is the max value of arm_joint. The vacuum_on is a bit
# for whether the vacuum is on.
CRVRobotType = Type("crv_robot")
Geom2DRobotEnvTypeFeatures[CRVRobotType] = [
    "x",
    "y",
    "theta",
    "base_radius",
    "arm_joint",
    "arm_length",
    "vacuum",
    "gripper_height",
    "gripper_width",
]
