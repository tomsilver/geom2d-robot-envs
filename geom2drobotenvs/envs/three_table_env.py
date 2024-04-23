"""Environment with blocks on three tables."""

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


class ThreeTableEnv(Geom2DRobotEnv):
    """Environment with blocks on three tables."""

    _robot_base_radius: ClassVar[float] = 0.36

    def _sample_initial_state(self) -> State:
        # Currently randomize just the order of blocks in the shelves. Can
        # also randomize the positions of stuff in the future.
        init_state_dict: Dict[Object, Dict[str, float]] = {}

        # Create the robot, initially facing right, at the center of the room.
        robot = CRVRobotType("robot")
        robot_x = (self._world_min_x + self._world_max_x) / 2.0
        init_state_dict[robot] = {
            "x": robot_x,
            "y": (self._world_min_y + self._world_max_y) / 2.0,
            "theta": 0.0,
            "base_radius": self._robot_base_radius,
            "arm_joint": self._robot_base_radius,  # arm is fully retracted
            "vacuum": 0.0,  # vacuum is off
        }

        # Common features for tables.
        common_table_feats = {
            "theta": 0.0,
            "static": True,
            "color_r": 139 / 255,
            "color_g": 139 / 255,
            "color_b": 139 / 255,
            "z_order": ZOrder.FLOOR.value,
        }

        # Create a table on the right.
        right_table = RectangleType("right_table")
        table_long_size = (self._world_max_x - self._world_min_x) / 5.0
        table_short_size = (self._world_max_y - self._world_min_y) / 7.0
        right_table_right_pad = table_long_size / 2
        right_table_x = self._world_max_x - (table_long_size + right_table_right_pad)
        right_table_y = (self._world_min_y + self._world_max_y - table_short_size) / 2.0
        init_state_dict[right_table] = {
            "x": right_table_x,
            "y": right_table_y,
            "width": table_long_size,
            "height": table_short_size,
            **common_table_feats,
        }

        # Create a table on the left.
        left_table = RectangleType("left_table")
        left_table_x =  self._world_min_x + (table_long_size - right_table_right_pad)
        left_table_y = (self._world_min_y + self._world_max_y - table_short_size) / 2.0
        init_state_dict[left_table] = {
            "x": left_table_x,
            "y": left_table_y,
            "width": table_long_size,
            "height": table_short_size,
            **common_table_feats,

        }

        # Create a table on the bottom.
        bottom_table = RectangleType("bottom_table")
        init_state_dict[bottom_table] = {
            "x": robot_x - (table_short_size / 2),
            "y": self._world_min_x + right_table_right_pad,
            "width": table_short_size,
            "height": table_long_size,
            **common_table_feats,
        }

        # Common features for table walls.
        common_table_wall_feats = {
            "theta": 0.0,
            "static": True,
            "color_r": 72 / 255,
            "color_g": 60 / 255,
            "color_b": 50 / 255,
            "z_order": ZOrder.SURFACE.value,
        }

        # Create right table walls.
        right_table_top_wall = RectangleType("right_table_top_wall")
        wall_thickness = table_long_size / 10.0
        init_state_dict[right_table_top_wall] = {
            "x": right_table_x,
            "y": right_table_y + table_short_size - wall_thickness,
            "width": table_long_size,
            "height": wall_thickness,
            **common_table_wall_feats,
        }
        right_table_bottom_wall = RectangleType("right_table_bottom_wall")
        init_state_dict[right_table_bottom_wall] = {
            "x": right_table_x,
            "y": right_table_y,
            "width": table_long_size,
            "height": wall_thickness,
            **common_table_wall_feats,
        }
        right_table_back_wall = RectangleType("right_table_back_wall")
        init_state_dict[right_table_back_wall] = {
            "x": right_table_x + table_long_size - wall_thickness,
            "y": right_table_y + wall_thickness,
            "width": wall_thickness,
            "height": table_short_size - 2 * wall_thickness,
            **common_table_wall_feats,
        }

        # Create left table walls.
        left_table_top_wall = RectangleType("left_table_top_wall")
        wall_thickness = table_long_size / 10.0
        init_state_dict[left_table_top_wall] = {
            "x": left_table_x,
            "y": left_table_y + table_short_size - wall_thickness,
            "width": table_long_size,
            "height": wall_thickness,
            **common_table_wall_feats,
        }
        left_table_bottom_wall = RectangleType("left_table_bottom_wall")
        init_state_dict[left_table_bottom_wall] = {
            "x": left_table_x,
            "y": left_table_y,
            "width": table_long_size,
            "height": wall_thickness,
            **common_table_wall_feats,
        }
        left_table_back_wall = RectangleType("left_table_back_wall")
        init_state_dict[left_table_back_wall] = {
            "x": left_table_x,
            "y": left_table_y + wall_thickness,
            "width": wall_thickness,
            "height": table_short_size - 2 * wall_thickness,
            **common_table_wall_feats,
        }


        # Create walls.
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
