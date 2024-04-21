"""Object types that are common across different environments."""

from relational_structs.structs import Type

# All geoms have an origin (x, y) and a rotation (in radians), and a bit
# indicating whether the geom is static (versus movable). They also have RGB.
Geom2DType = Type(
    "geom2d", ["x", "y", "theta", "is_static", "color_r", "color_g", "color_b"]
)
# Specific geom types.
RectangleType = Type(
    "rectangle", Geom2DType.feature_names + ["width", "height"], parent=Geom2DType
)
# A robot with a circle base, a rectangle arm, and a vacuum rectangle gripper.
# The (x, y, theta) are for the center of the robot base circle. The base_radius
# is for that circle. The arm_joint is a distance between the center and the
# gripper. The vacuum_on is a bit for whether the vacuum is on.
CRVRobotType = Type(
    "crv_robot", ["x", "y", "theta", "base_radius", "arm_joint", "vacuum_on"]
)