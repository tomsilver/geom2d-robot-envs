"""Data structures."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import matplotlib.pyplot as plt
from tomsgeoms2d.structs import Geom2D


class ZOrder(Enum):
    """Used for collision checking."""

    FLOOR = 1  # collides with things on the floor
    SURFACE = 2  # collides with things at the table surface level
    ALL = 100  # collides with everyone (except NONE)
    NONE = 101  # collides with nothing


def z_orders_may_collide(z0: ZOrder, z1: ZOrder) -> bool:
    """Defines the semantics of ZOrder collisions."""
    if ZOrder.NONE in (z0, z1):
        return False
    if ZOrder.ALL in (z0, z1):
        return True
    return z0 == z1


@dataclass(frozen=True)
class Body2D:
    """A body consists a geom, a z order (for collision checks), and rendering
    kwargs (for visualization)."""

    geom: Geom2D
    z_order: ZOrder
    rendering_kwargs: Dict
    name: str = "root"

    def plot(self, ax: plt.Axes) -> None:
        """Render the body in matplotlib."""
        self.geom.plot(ax=ax, zorder=self.z_order.value, **self.rendering_kwargs)


@dataclass(frozen=True)
class MultiBody2D:
    """A container for bodies."""

    name: str
    bodies: List[Body2D]

    def plot(self, ax: plt.Axes) -> None:
        """Render the bodies in matplotlib."""
        for body in self.bodies:
            body.plot(ax)

    def get_body(self, name: str) -> Body2D:
        """Retrieve a body by its name."""
        for body in self.bodies:
            if body.name == name:
                return body
        raise ValueError(f"Multibody {self.name} does not contain body {name}")
