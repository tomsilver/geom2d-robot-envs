"""Environment with blocks on three tables."""

from typing import ClassVar, Dict

import numpy as np
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

    _robot_base_radius: ClassVar[float] = 0.4
    _num_blocks: ClassVar[int] = 6

    def _sample_initial_state(self) -> State:
        # Currently nothing is randomized; this will change in the future.
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
            "arm_length": 6 * self._robot_base_radius,
            "vacuum": 0.0,  # vacuum is off
            "gripper_height": 0.7,
            "gripper_width": 0.1,
        }

        # Common features for tables.
        common_table_feats = {
            "theta": 0.0,
            "static": True,
            "color_r": 200 / 255,
            "color_g": 200 / 255,
            "color_b": 200 / 255,
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
        left_table_x = self._world_min_x + (table_long_size - right_table_right_pad)
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
        bottom_table_x = robot_x - (table_short_size / 2)
        bottom_table_y = self._world_min_x + right_table_right_pad
        init_state_dict[bottom_table] = {
            "x": bottom_table_x,
            "y": bottom_table_y,
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
        wall_thickness = table_long_size / 10.0
        right_table_top_wall = RectangleType("right_table_top_wall")
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

        # Create bottom table walls.
        bottom_table_left_wall = RectangleType("bottom_table_left_wall")
        init_state_dict[bottom_table_left_wall] = {
            "x": bottom_table_x,
            "y": bottom_table_y,
            "width": wall_thickness,
            "height": table_long_size,
            **common_table_wall_feats,
        }
        bottom_table_right_wall = RectangleType("bottom_table_right_wall")
        init_state_dict[bottom_table_right_wall] = {
            "x": bottom_table_x + table_short_size - wall_thickness,
            "y": bottom_table_y,
            "width": wall_thickness,
            "height": table_long_size,
            **common_table_wall_feats,
        }
        bottom_table_bottom_wall = RectangleType("bottom_table_bottom_wall")
        init_state_dict[bottom_table_bottom_wall] = {
            "x": bottom_table_x + wall_thickness,
            "y": bottom_table_y,
            "width": table_short_size - 2 * wall_thickness,
            "height": wall_thickness,
            **common_table_wall_feats,
        }

        # Create blocks that need to be moved around.
        common_block_feats = {
            "theta": 0.0,
            "static": False,
            "color_r": 173 / 255,  # blue
            "color_g": 216 / 255,
            "color_b": 230 / 255,
            "z_order": ZOrder.SURFACE.value,
        }
        block_max_thickness = (table_long_size - wall_thickness) / self._num_blocks
        block_thickness = 0.5 * block_max_thickness  # conservative
        block_max_long_size = table_short_size - 2 * wall_thickness
        largest_block_long_size = 0.8 * block_max_long_size  # conservative
        smallest_block_long_size = largest_block_long_size / 2
        block_long_sizes = np.linspace(
            largest_block_long_size,
            smallest_block_long_size,
            self._num_blocks,
            endpoint=True,
        )
        for block_num, block_size in enumerate(block_long_sizes):
            # For now, all blocks are initialized on the right table.
            table_back_x = right_table_x + table_long_size - wall_thickness
            pad_x = 0.75 * block_thickness
            block_x = table_back_x - (block_num + 1) * (block_thickness + pad_x)
            block_y = right_table_y + table_short_size / 2 - block_size / 2
            init_state_dict[RectangleType(f"block{block_num}")] = {
                "x": block_x,
                "y": block_y,
                "width": block_thickness,
                "height": block_size,
                **common_block_feats,
            }

        # Create room walls.
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
