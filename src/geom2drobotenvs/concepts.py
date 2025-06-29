"""Boolean concepts that may be useful for goals, high-level planning, etc."""

from relational_structs import Object, ObjectCentricState

from geom2drobotenvs.object_types import RectangleType
from geom2drobotenvs.structs import MultiBody2D
from geom2drobotenvs.utils import rectangle_object_to_geom


def is_inside(
    state: ObjectCentricState,
    inner: Object,
    outer: Object,
    static_object_cache: dict[Object, MultiBody2D],
) -> bool:
    """Checks if the inner object is completely inside the outer one.

    Only rectangles are currently supported.
    """
    inner_geom = rectangle_object_to_geom(state, inner, static_object_cache)
    outer_geom = rectangle_object_to_geom(state, outer, static_object_cache)
    for x, y in inner_geom.vertices:
        if not outer_geom.contains_point(x, y):
            return False
    return True


def is_on(
    state: ObjectCentricState,
    top: Object,
    bottom: Object,
    static_object_cache: dict[Object, MultiBody2D],
    tol: float = 1e-1,
) -> bool:
    """Checks top object is completely on the bottom one.

    Only rectangles are currently supported.

    Assumes that "up" is positive y.
    """
    top_geom = rectangle_object_to_geom(state, top, static_object_cache)
    bottom_geom = rectangle_object_to_geom(state, bottom, static_object_cache)
    # The bottom-most vertices of top_geom should be contained within the bottom
    # geom when those vertices are offset by tol.
    sorted_vertices = sorted(top_geom.vertices, key=lambda v: v[1])
    for x, y in sorted_vertices[:2]:
        offset_y = y - tol
        if not bottom_geom.contains_point(x, offset_y):
            return False
    return True


def is_movable_rectangle(state: ObjectCentricState, obj: Object) -> bool:
    """Checks if an object is a movable rectangle."""
    return obj.is_instance(RectangleType) and state.get(obj, "static") < 0.5
