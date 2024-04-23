"""Shelf world environment."""

from typing import ClassVar, Dict

from relational_structs.structs import Object, State
from relational_structs.utils import create_state_from_dict

from geom2drobotenvs.envs.base_env import Geom2DRobotEnv
from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.structs import ZOrder
from geom2drobotenvs.utils import (
    CRVRobotActionSpace,
    create_walls_from_world_boundaries,
)


class ShelfWorldEnv(Geom2DRobotEnv):
    """Shelf world environment."""

    _robot_base_radius: ClassVar[float] = 0.36

    def _sample_initial_state(self) -> State:
        # Coming soon: randomization.
        init_state_dict: Dict[Object, Dict[str, float]] = {}
        robot = CRVRobotType("robot")
        init_state_dict[robot] = {
            "x": (self._world_min_x + self._world_max_x) / 2.0,  # center of room
            "y": (self._world_min_y + self._world_max_y) / 2.0,
            "theta": 0.0,  # facing right
            "base_radius": self._robot_base_radius,
            "arm_joint": self._robot_base_radius,  # arm is fully retracted
            "vacuum": 0.0,  # vacuum is off
        }
        right_table = RectangleType("right_table")
        right_table_width = (self._world_max_x - self._world_min_x) / 10.0
        right_table_height = (self._world_max_y - self._world_min_y) / 3.0
        right_table_right_pad = right_table_width / 2
        init_state_dict[right_table] = {
            # Origin is bottom left hand corner.
            "x": self._world_max_x - (right_table_width + right_table_right_pad),
            "y": (self._world_min_y + self._world_max_y - right_table_height) / 2.0,
            "width": right_table_width,
            "height": right_table_height,
            "theta": 0.0,
            "static": True,  # table can't move
            "color_r": 139 / 255,  # brown
            "color_g": 39 / 255,
            "color_b": 19 / 255,
            "z_order": ZOrder.FLOOR.value,
        }
        assert isinstance(self.action_space, CRVRobotActionSpace)
        min_dx, min_dy = self.action_space.low[:2]
        max_dx, max_dy = self.action_space.high[:2]
        wall_state_dict = create_walls_from_world_boundaries(
            self._world_min_x,
            self._world_max_x,
            self._world_min_y,
            self._world_max_y,
            min_dx,
            max_dx,
            min_dy,
            max_dy,
        )
        init_state_dict.update(wall_state_dict)
        # Finalize state.
        return create_state_from_dict(init_state_dict)
