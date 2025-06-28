"""Environment where a block must be placed on an obstructed target."""

from dataclasses import dataclass

import numpy as np
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from geom2drobotenvs.envs.base_env import Geom2DRobotEnv, Geom2DRobotEnvSpec
from geom2drobotenvs.object_types import (
    CRVRobotType,
    Geom2DRobotEnvTypeFeatures,
    RectangleType,
)
from geom2drobotenvs.structs import ZOrder
from geom2drobotenvs.utils import (
    CRVRobotActionSpace,
    SE2Pose,
    create_walls_from_world_boundaries,
)


@dataclass(frozen=True)
class Obstruction2DEnvSpec(Geom2DRobotEnvSpec):
    """Scene specification for Obstruction2DEnv()."""

    # World boundaries. Standard coordinate frame with (0, 0) in bottom left.
    world_min_x: float = 0.0
    world_max_x: float = (1 + np.sqrt(5)) / 2  # golden ratio :)
    world_min_y: float = 0.0
    world_max_y: float = 1.0

    # Action space parameters.
    min_dx: float = -5e-2
    max_dx: float = 5e-2
    min_dy: float = -5e-2
    max_dy: float = 5e-2
    min_dtheta: float = -np.pi / 16
    max_dtheta: float = np.pi / 16
    min_darm: float = -1e-1
    max_darm: float = 1e-1
    min_vac: float = 0.0
    max_vac: float = 1.0

    # Robot hyperparameters.
    robot_base_radius: float = 0.1
    robot_arm_length: float = 6 * robot_base_radius
    robot_gripper_height: float = 0.07
    robot_gripper_width: float = 0.01

    # Table hyperparameters.
    table_rgb: tuple[float, float, float] = (0.75, 0.75, 0.75)
    table_height: float = 0.1
    table_width: float = world_max_x - world_min_x
    # The table pose is defined relative to the bottom left hand corner.
    table_pose: SE2Pose = SE2Pose(world_min_x, world_min_y, 0.0)

    # Target surface hyperparameters.
    target_surface_rgb: tuple[float, float, float] = (0.75, 0.1, 0.1)
    target_surface_y: float = table_pose.y
    target_surface_theta: float = table_pose.theta
    target_surface_height: float = table_height

    # Target block hyperparameters.
    target_block_rgb: tuple[float, float, float] = (0.75, 0.1, 0.1)
    target_block_y: float = table_pose.y + table_height
    target_block_theta: float = table_pose.theta
    target_block_height: float = robot_base_radius

    # For rendering.
    render_dpi: int = 200


class Obstruction2DEnv(Geom2DRobotEnv):
    """Environment where a block must be placed on an obstructed target."""

    def __init__(self, spec: Obstruction2DEnvSpec = Obstruction2DEnvSpec()) -> None:
        super().__init__(spec)
        self._spec: Obstruction2DEnvSpec = spec  # for type checking

    def _sample_initial_state(self) -> ObjectCentricState:
        # TODO randomize.
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the robot at the top center of the world.
        robot = CRVRobotType("robot")
        robot_x = (self._spec.world_min_x + self._spec.world_max_x) / 2.0
        robot_y = self._spec.world_max_y - self._spec.robot_base_radius
        robot_theta = -np.pi / 2  # facing down
        init_state_dict[robot] = {
            "x": robot_x,
            "y": robot_y,
            "theta": robot_theta,
            "base_radius": self._spec.robot_base_radius,
            "arm_joint": self._spec.robot_base_radius,  # arm is fully retracted
            "arm_length": self._spec.robot_arm_length,
            "vacuum": 0.0,  # vacuum is off
            "gripper_height": self._spec.robot_gripper_height,
            "gripper_width": self._spec.robot_gripper_width,
        }

        # Create the table.
        table = RectangleType("table")
        init_state_dict[table] = {
            "x": self._spec.table_pose.x,
            "y": self._spec.table_pose.y,
            "theta": self._spec.table_pose.theta,
            "width": self._spec.table_width,
            "height": self._spec.table_height,
            "static": True,
            "color_r": self._spec.table_rgb[0],
            "color_g": self._spec.table_rgb[1],
            "color_b": self._spec.table_rgb[2],
            "z_order": ZOrder.ALL.value,
        }

        # Create the target surface.
        target_surface = RectangleType("target_surface")
        # TODO randomize x and width.
        init_state_dict[target_surface] = {
            "x": (self._spec.world_max_x + self._spec.world_min_x) / 2,  # TODO
            "y": self._spec.target_surface_y,
            "theta": self._spec.target_surface_theta,
            "width": self._spec.robot_base_radius * 2.5,  # TODO
            "height": self._spec.target_surface_height,
            "static": True,
            "color_r": self._spec.target_surface_rgb[0],
            "color_g": self._spec.target_surface_rgb[1],
            "color_b": self._spec.target_surface_rgb[2],
            "z_order": ZOrder.NONE.value,
        }

        # Create target block.
        target_block = RectangleType("target_block")
        # TODO randomize x and width.
        init_state_dict[target_block] = {
            "x": (self._spec.world_max_x + self._spec.world_min_x) / 2,  # TODO
            "y": self._spec.target_block_y,
            "theta": self._spec.target_block_theta,
            "width": self._spec.robot_base_radius * 2,  # TODO
            "height": self._spec.target_block_height,
            "static": True,
            "color_r": self._spec.target_block_rgb[0],
            "color_g": self._spec.target_block_rgb[1],
            "color_b": self._spec.target_block_rgb[2],
            "z_order": ZOrder.ALL.value,
        }

        # Create room walls.
        assert isinstance(self.action_space, CRVRobotActionSpace)
        min_dx, min_dy = self.action_space.low[:2]
        max_dx, max_dy = self.action_space.high[:2]
        wall_state_dict = create_walls_from_world_boundaries(
            self._spec.world_min_x,
            self._spec.world_max_x,
            self._spec.world_min_y,
            self._spec.world_max_y,
            min_dx,
            max_dx,
            min_dy,
            max_dy,
        )
        init_state_dict.update(wall_state_dict)

        # Finalize state.
        return create_state_from_dict(init_state_dict, Geom2DRobotEnvTypeFeatures)
