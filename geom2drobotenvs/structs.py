"""Data structures."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import matplotlib.pyplot as plt
from tomsgeoms2d.structs import Geom2D


class ZOrder(Enum):
    """Used for collision checking."""

    ALL = 1  # collides with everyone
    FLOOR = 2  # collides with things on the floor
    SURFACE = 3  # collides with things at the table surface level


def z_orders_may_collide(z0: ZOrder, z1: ZOrder) -> bool:
    """Defines the semantics of ZOrder collisions."""
    if ZOrder.ALL in (z0, z1):
        return True
    return z0 == z1


@dataclass(frozen=True)
class Body2D:
    """A body consists of one or more geoms, which each have a z order and
    rendering kwargs (e.g., facecolor).

    The color is for rendering and the z order is for collision
    checking.
    """

    geoms: List[Geom2D]
    z_orders: List[ZOrder]
    rendering_kwargs: List[Dict]

    def __postinit__(self) -> None:
        assert len(self.geoms) == len(self.z_orders) == len(self.rendering_kwargs)

    def plot(self, ax: plt.Axes) -> None:
        """Render the body in matplotlib."""
        for geom, kwargs in zip(self.geoms, self.rendering_kwargs):
            geom.plot(ax=ax, **kwargs)
