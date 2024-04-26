"""Boolean concepts that may be useful for goals, high-level planning, etc."""

from typing import Dict

from relational_structs import Object, State
from tomsgeoms2d.structs import Rectangle

from geom2drobotenvs.object_types import RectangleType
from geom2drobotenvs.structs import MultiBody2D
from geom2drobotenvs.utils import object_to_multibody2d, rectangle_object_to_geom


def is_inside(
    state: State,
    inner: Object,
    outer: Object,
    static_object_cache: Dict[Object, MultiBody2D],
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


def is_movable_rectangle(state: State, obj: Object) -> bool:
    """Checks if an object is a movable rectangle."""
    return obj.is_instance(RectangleType) and state.get(obj, "static") < 0.5
