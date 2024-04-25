"""Boolean concepts that may be useful for goals, high-level planning, etc."""

from typing import Dict

from relational_structs.structs import Object, State
from tomsgeoms2d.structs import Rectangle

from geom2drobotenvs.structs import MultiBody2D
from geom2drobotenvs.utils import object_to_multibody2d


def is_inside(
    state: State,
    inner: Object,
    outer: Object,
    static_object_cache: Dict[Object, MultiBody2D],
) -> bool:
    """Checks if the inner object is completely inside the outer one.

    Only rectangles are currently supported.
    """
    inner_mb = object_to_multibody2d(inner, state, static_object_cache)
    assert len(inner_mb.bodies) == 1
    inner_geom = inner_mb.bodies[0].geom
    assert isinstance(inner_geom, Rectangle)
    outer_mb = object_to_multibody2d(outer, state, static_object_cache)
    assert len(outer_mb.bodies) == 1
    outer_geom = outer_mb.bodies[0].geom
    assert isinstance(outer_geom, Rectangle)
    for x, y in inner_geom.vertices:
        if not outer_geom.contains_point(x, y):
            return False
    return True
